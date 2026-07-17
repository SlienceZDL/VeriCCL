from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import time
from typing import Callable, Optional, Tuple

from vericcl.artifacts.hashing import candidate_signature
from vericcl.artifacts.layout import RunLayout, create_run_layout
from vericcl.artifacts.summary import build_run_summary
from vericcl.artifacts.writer import (
    CandidateArtifact,
    read_schedule_sidecar,
    write_candidate_artifact,
    atomic_write_text,
    write_final_alias,
    write_resolved_input,
    write_run_summary,
)
from vericcl.composer import compose
from vericcl.errors import SemanticError
from vericcl.input.loader import resolve_inputs
from vericcl.input.models import ResolvedInput
from vericcl.planner.build import build_plan
from vericcl.semantics.atom import Schedule
from vericcl.solver.model import SolveCandidate, SolveRequest
from vericcl.solver.orchestrator import solve
from vericcl.topology.loader import load_topology
from vericcl.tuning.engine import (
    CandidateAssessment,
    OnlinePerformance,
    TuningContext,
    TuningHistoryEntry,
    tune,
)
from vericcl.tuning.model import TuningOverlay
from vericcl.verification.model import (
    CheckResult,
    ValidationStatus,
)
from vericcl.verification.online.pipeline import (
    OnlineContext,
    OnlineStageStatus,
    OnlineValidationResult,
    attach_online_result_to_tuning_context,
    run_online_validation,
)
from vericcl.verification.online.calibration import (
    CalibrationResult,
    apply_calibration_to_topology,
)
from vericcl.verification.pipeline import (
    VerificationOutcome,
    validate_and_lower_candidate,
    verify_candidate_outcome,
)
from vericcl.xml.lower import XmlArtifact


_monotonic = time.monotonic


OnlineContextFactory = Callable[
    [XmlArtifact, Schedule, ResolvedInput, Path, Path, bool, float, bool],
    OnlineContext,
]


@dataclass(frozen=True)
class RunContext:
    topology_path: Path
    sketch_path: Path
    atom_path: Path
    output_base: Path
    run_id: str
    xml_path: Optional[Path] = None
    sidecar_path: Optional[Path] = None
    online: bool = False
    tune: bool = False
    timeout_s: Optional[float] = None
    solver_version: str = "unknown"
    model_version: str = "1"
    environment_signature: str = "unknown"
    online_context_factory: Optional[OnlineContextFactory] = None

    def __post_init__(self) -> None:
        for field in (
            "topology_path",
            "sketch_path",
            "atom_path",
            "output_base",
            "xml_path",
            "sidecar_path",
        ):
            value = getattr(self, field)
            if value is not None:
                try:
                    object.__setattr__(self, field, Path(value))
                except TypeError as error:
                    raise SemanticError(
                        "run context {} must be path-like".format(field)
                    ) from error
        if not isinstance(self.run_id, str) or not self.run_id:
            raise SemanticError("run context run_id must be a non-empty string")
        if (self.xml_path is None) != (self.sidecar_path is None):
            raise SemanticError(
                "run context XML and schedule sidecar must be provided together"
            )
        for field in ("online", "tune"):
            if not isinstance(getattr(self, field), bool):
                raise SemanticError(
                    "run context {} must be a boolean".format(field)
                )
        if self.online_context_factory is not None and not callable(
            self.online_context_factory
        ):
            raise SemanticError(
                "run context online_context_factory must be callable"
            )
        if self.timeout_s is not None:
            if (
                isinstance(self.timeout_s, bool)
                or not isinstance(self.timeout_s, (int, float))
                or self.timeout_s <= 0.0
            ):
                raise SemanticError("run context timeout_s must be positive")
        for field in (
            "solver_version",
            "model_version",
            "environment_signature",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise SemanticError(
                    "run context {} must be a non-empty string".format(field)
                )


@dataclass(frozen=True)
class RunArtifacts:
    layout: RunLayout
    candidates: Tuple[CandidateArtifact, ...]
    final_candidate_id: Optional[str]
    final_xml: Optional[Path]
    final_report: Optional[Path]
    status: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.layout, RunLayout):
            raise SemanticError("run artifacts layout is invalid")
        candidates = tuple(self.candidates)
        if not all(isinstance(item, CandidateArtifact) for item in candidates):
            raise SemanticError("run artifacts candidates are invalid")
        object.__setattr__(self, "candidates", candidates)
        if self.final_candidate_id is not None and self.final_candidate_id not in {
            item.candidate_id for item in candidates
        }:
            raise SemanticError("run artifacts final candidate does not exist")


class _Deadline:
    def __init__(self, timeout_s: float, started_at: float) -> None:
        self._deadline = float(started_at) + float(timeout_s)

    def check(self, stage: str) -> None:
        if _monotonic() > self._deadline:
            raise TimeoutError(
                "workflow wall-clock budget expired during {}".format(stage)
            )

    def remaining(self) -> float:
        return max(0.0, self._deadline - _monotonic())


@dataclass(frozen=True)
class _CandidateRecord:
    candidate: SolveCandidate
    schedule: Schedule
    outcome: VerificationOutcome
    topology: object
    overlay: Optional[TuningOverlay]
    tuning_strategy: dict
    accepted: bool
    rejection_reason: Optional[str]


def _timeout(context: RunContext, configured: int) -> float:
    return float(configured if context.timeout_s is None else context.timeout_s)


def _hierarchy_plan(plan) -> dict:
    return {
        "nodes": tuple(
            {
                "node_id": node.node_id,
                "stage_id": node.stage_id,
                "operator": node.local_collective.kind.value,
                "communication_group": node.communication_group,
                "dual_of_node_id": node.dual_of_node_id,
            }
            for node in plan.nodes
        ),
        "edges": tuple(
            {
                "producer_id": edge.producer_id,
                "consumer_id": edge.consumer_id,
            }
            for edge in plan.edges
        ),
    }


def _applied_strategies(inputs) -> dict:
    values = inputs.strategies
    return {
        "hierarchy": values.hierarchy,
        "symmetry": values.symmetry,
        "shortest_paths": values.shortest_paths,
        "batching": values.batching,
        "constructive_trees": values.constructive_trees,
        "milp": values.milp,
    }


def _offline_valid(outcome: VerificationOutcome) -> bool:
    report = outcome.report
    return (
        report.overall_status is ValidationStatus.VALID
        and report.bdd.status is ValidationStatus.VALID
        and report.simulation.status is ValidationStatus.VALID
    )


def _rejection_reason(outcome: VerificationOutcome) -> Optional[str]:
    if _offline_valid(outcome):
        return None
    report = outcome.report
    for dimension in (
        "input",
        "semantic",
        "state",
        "topology",
        "timing",
        "resource",
        "buffer",
        "endpoint",
        "deadlock",
        "xml",
        "bdd",
        "simulation",
    ):
        result = getattr(report, dimension)
        if result.status is not ValidationStatus.VALID:
            return result.code
    return "offline_validation_failed"


def _lane_evidence(lane) -> Optional[dict]:
    if lane is None:
        return None
    return {
        "src_rank": lane.src_rank,
        "dst_rank": lane.dst_rank,
        "channel": lane.channel,
    }


def _bottleneck_evidence(item) -> dict:
    return {
        "transfer_id": item.transfer_id,
        "stage_id": item.stage_id,
        "endpoint_type": item.endpoint_type.value,
        "atom_ids": item.atom_ids,
        "flow_ids": item.flow_ids,
        "rank": item.rank,
        "tb_id": item.tb_id,
        "step_index": item.step_index,
        "iteration": item.iteration,
        "lane": _lane_evidence(item.lane),
        "wait_class": item.wait_class.value,
        "duration_us": item.duration_us,
        "ordering_confident": item.ordering_confident,
    }


def _trace_analysis_evidence(analysis) -> Optional[dict]:
    if analysis is None:
        return None
    return {
        "intervals": tuple(
            {
                "transfer_id": interval.transfer_id,
                "iteration": interval.iteration,
                "physical_start_us": interval.physical_start_us,
                "physical_end_us": interval.physical_end_us,
                "start_uncertainty_us": (
                    interval.physical_start.uncertainty_us
                ),
                "end_uncertainty_us": interval.physical_end.uncertainty_us,
                "endpoint_order_uncertain": (
                    interval.endpoint_order_uncertain
                ),
            }
            for interval in analysis.intervals
        ),
        "step_waits": tuple(
            {
                "transfer_id": step.transfer_id,
                "atom_ids": step.record.atom_ids,
                "flow_ids": step.record.flow_ids,
                "rank": step.record.rank,
                "tb_id": step.record.tb_id,
                "step_index": step.record.step_index,
                "iteration": step.record.iteration,
                "endpoint_type": step.record.endpoint_type.value,
                "lane": _lane_evidence(step.record.lane),
                "semantic_ready_us": step.semantic_ready_us,
                "semantic_ready_uncertainty_us": (
                    step.semantic_ready.uncertainty_us
                ),
                "head_of_line_wait_us": (
                    step.waits.head_of_line_wait_us
                ),
                "dependency_wait_us": step.waits.dependency_wait_us,
                "peer_resource_wait_us": (
                    step.waits.peer_resource_wait_us
                ),
                "transfer_duration_us": (
                    step.waits.transfer_duration_us
                ),
                "ordering_confident": step.ordering_confident,
            }
            for step in analysis.step_waits
        ),
        "bottlenecks": tuple(
            _bottleneck_evidence(item) for item in analysis.bottlenecks
        ),
        "uncertain_comparisons": analysis.uncertain_comparisons,
        "tuning_eligible": analysis.tuning_eligible,
    }


def _online_evidence(result: OnlineValidationResult) -> dict:
    history = result.release_history
    calibration = result.calibration
    return {
        "preflight_status": result.preflight_status.value,
        "calibration_status": result.calibration_status.value,
        "release_status": result.release_status.value,
        "online_operator_validation": (
            result.online_operator_validation.value
        ),
        "failure_code": result.failure_code,
        "failure_message": result.failure_message,
        "release_rounds": (
            ()
            if history is None
            else tuple(
                {
                    "sample_count": value.sample_count,
                    "median_us": value.median_us,
                    "p95_us": value.p95_us,
                    "mean_us": value.mean_us,
                    "population_standard_deviation_us": (
                        value.population_standard_deviation_us
                    ),
                    "coefficient_of_variation": (
                        value.coefficient_of_variation
                    ),
                    "stable": value.stable,
                }
                for value in history.rounds
            )
        ),
        "calibration": (
            None
            if calibration is None
            else {
                "link_class": calibration.request.link_class,
                "slice_size_bytes": calibration.request.slice_size_bytes,
                "benchmark_size_bytes": (
                    calibration.request.benchmark_size_bytes
                ),
                "max_calibration_channels": (
                    calibration.request.max_calibration_channels
                ),
                "cache_hit_concurrencies": (
                    calibration.cache_hit_concurrencies
                ),
                "stable": calibration.stable,
                "skipped_reason": calibration.skipped_reason,
                "points": tuple(
                    {
                        "concurrency": point.concurrency,
                        "median_us": (
                            point.duration_statistics.median_us
                        ),
                        "p95_us": point.duration_statistics.p95_us,
                        "mean_us": point.duration_statistics.mean_us,
                        "population_standard_deviation_us": (
                            point.duration_statistics.population_standard_deviation_us
                        ),
                        "coefficient_of_variation": (
                            point.duration_statistics.coefficient_of_variation
                        ),
                        "stable": point.stable,
                        "full_wave_count": point.full_wave_count,
                        "tail_transfer_count": (
                            point.tail_transfer_count
                        ),
                    }
                    for point in calibration.points
                ),
                "curve": (
                    None
                    if calibration.curve is None
                    else {
                        "alpha_us": calibration.curve.alpha_us,
                        "invbw_us": calibration.curve.invbw_us,
                        "bandwidth_bytes_per_us": {
                            str(concurrency): bandwidth
                            for concurrency, bandwidth in (
                                calibration.curve.bandwidth_bytes_per_us.items()
                            )
                        },
                    }
                ),
            }
        ),
        "trace_rank_files": tuple(
            str(path) for path in result.trace_rank_files
        ),
        "trace_clock_uncertainty_us": (
            result.trace_clock_uncertainty_us
        ),
        "trace_analysis": _trace_analysis_evidence(result.trace_analysis),
        "tuning_evidence": (
            None
            if result.tuning_evidence is None
            else {
                "wait_us_by_transfer": dict(
                    result.tuning_evidence.wait_us_by_transfer
                ),
                "bottleneck_priorities": tuple(
                    _bottleneck_evidence(item)
                    for item in result.tuning_evidence.bottleneck_priorities
                ),
            }
        ),
        "requires_resolve": result.requires_resolve,
        "online_tuning_allowed": result.online_tuning_allowed,
    }


def _with_online_result(
    outcome: VerificationOutcome,
    result: OnlineValidationResult,
) -> VerificationOutcome:
    passed = (
        result.online_operator_validation is OnlineStageStatus.PASSED
        and result.release_status is OnlineStageStatus.PASSED
        and result.failure_code is None
    )
    online = CheckResult(
        dimension="online",
        status=(
            ValidationStatus.VALID if passed else ValidationStatus.FAILED
        ),
        code=(
            "online_validation_passed"
            if passed
            else result.failure_code or "online_validation_failed"
        ),
        message=(
            "online operator validation passed"
            if passed
            else result.failure_message or "online operator validation failed"
        ),
        evidence=_online_evidence(result),
    )
    return replace(
        outcome,
        report=replace(outcome.report, online=online),
    )


def _with_online_failure(
    outcome: VerificationOutcome,
    code: str,
    message: str,
) -> VerificationOutcome:
    return replace(
        outcome,
        report=replace(
            outcome.report,
            online=CheckResult(
                dimension="online",
                status=ValidationStatus.FAILED,
                code=code,
                message=message,
                evidence={},
            ),
        ),
    )


def _run_online_candidate(
    *,
    candidate_id: str,
    schedule: Schedule,
    outcome: VerificationOutcome,
    inputs,
    layout: RunLayout,
    factory: OnlineContextFactory,
    tuning_requested: bool,
    deadline: _Deadline,
    calibrate: bool = True,
) -> tuple:
    if outcome.artifact is None or not outcome.report.runtime_compatible:
        return (
            _with_online_failure(
                outcome,
                "online_runtime_incompatible",
                "online validation requires a runtime-compatible XML",
            ),
            None,
        )
    remaining = deadline.remaining()
    if remaining <= 0.0:
        raise TimeoutError(
            "workflow wall-clock budget expired before online validation"
        )
    token = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:12]
    candidate_traces = layout.traces / "candidate-{}-{}".format(
        token,
        "calibration" if calibrate else "operator",
    )
    candidate_traces.mkdir()
    xml_path = candidate_traces / "online-input.xml"
    atomic_write_text(xml_path, outcome.artifact.xml_text)
    online_context = factory(
        outcome.artifact,
        schedule,
        inputs,
        xml_path,
        candidate_traces,
        tuning_requested,
        remaining,
        calibrate,
    )
    result = run_online_validation(online_context)
    deadline.check("online validation")
    return _with_online_result(outcome, result), result


def _applied_calibration(result: OnlineValidationResult) -> CalibrationResult:
    calibration = result.calibration
    if (
        not result.requires_resolve
        or calibration is None
        or not calibration.stable
        or calibration.curve is None
        or calibration.skipped_reason is not None
    ):
        raise SemanticError("online resolve requires stable calibration")
    return CalibrationResult(
        request=calibration.request,
        points=calibration.points,
        curve=calibration.curve,
        skipped_reason=None,
    )


def _merge_applied_calibration(
    calibration_result: OnlineValidationResult,
    operator_result: OnlineValidationResult,
    *,
    requires_resolve: bool = False,
) -> OnlineValidationResult:
    return replace(
        operator_result,
        calibration_status=OnlineStageStatus.PASSED,
        calibration=calibration_result.calibration,
        requires_resolve=requires_resolve,
        online_tuning_allowed=(
            False
            if requires_resolve
            else operator_result.online_tuning_allowed
        ),
        tuning_evidence=(
            None if requires_resolve else operator_result.tuning_evidence
        ),
    )


def _global_schedule(plan, candidate: SolveCandidate) -> Schedule:
    return compose(
        plan,
        {node.node_id: candidate for node in plan.nodes},
    )


def _select_final(
    candidates: Tuple[SolveCandidate, ...],
    outcomes: Tuple[VerificationOutcome, ...],
    selected_candidate_id: Optional[str],
) -> Optional[str]:
    eligible = tuple(
        candidate.candidate_id
        for candidate, outcome in zip(candidates, outcomes)
        if _offline_valid(outcome)
    )
    if selected_candidate_id in eligible:
        return selected_candidate_id
    return eligible[0] if eligible else None


def _tuned_candidate(
    initial: SolveCandidate,
    entry: TuningHistoryEntry,
) -> SolveCandidate:
    schedule = entry.schedule
    makespan = max(
        (transfer.ed_time for transfer in schedule.transfers),
        default=0.0,
    )
    operation_count = len(schedule.transfers)
    metrics = replace(
        initial.metrics,
        status=initial.metrics.status,
        objective_values=(
            entry.simulation_time_us
            if entry.simulation_time_us is not None
            else makespan,
        ),
        best_bound=0.0,
        mip_gap=0.0,
        within_requested_gap=False,
        solve_time_s=0.0,
        model_count=0,
        operation_count=operation_count,
        hop_count=operation_count,
        makespan_us=makespan,
        maximum_normalized_resource_load=makespan,
        solver_name="vericcl-tuner",
        termination_reason="tuning_candidate",
    )
    restrictions = tuple(
        sorted(set(initial.restrictions) | {"tuning_candidate_space"})
    )
    return SolveCandidate(
        candidate_id=entry.candidate_id,
        node_schedules={"global": schedule},
        objective_mode=initial.objective_mode,
        channel_count=(
            entry.overlay.channel_count
            if entry.overlay is not None
            and entry.overlay.channel_count is not None
            else initial.channel_count
        ),
        metrics=metrics,
        selected_best=entry.selected_best,
        proven_optimal=False,
        search_space_restricted=True,
        restrictions=restrictions,
        parent_candidate_id=entry.parent_candidate_id,
    )


def _tuning_records(
    initial: SolveCandidate,
    schedule: Schedule,
    initial_outcome: VerificationOutcome,
    inputs,
    topology,
    deadline: _Deadline,
    *,
    online_result: Optional[OnlineValidationResult] = None,
    online_factory: Optional[OnlineContextFactory] = None,
    layout: Optional[RunLayout] = None,
) -> tuple:
    remaining = deadline.remaining()
    if remaining <= 0.0:
        raise TimeoutError("workflow wall-clock budget expired before tuning")
    tuning_context = TuningContext(
        inputs=inputs,
        topology=topology,
        initial_schedule=schedule,
        max_iterations=inputs.hyperparameters.max_tuning_iterations,
        timeout_s=remaining,
    )

    def assessment(proposal) -> CandidateAssessment:
        if proposal.candidate_id == initial.candidate_id:
            outcome = initial_outcome
            candidate_online_result = online_result
        else:
            outcome = validate_and_lower_candidate(
                proposal.schedule,
                inputs,
                topology,
            )
            candidate_online_result = None
            if online_result is not None:
                assert online_factory is not None and layout is not None
                outcome, candidate_online_result = _run_online_candidate(
                    candidate_id=proposal.candidate_id,
                    schedule=proposal.schedule,
                    outcome=outcome,
                    inputs=inputs,
                    layout=layout,
                    factory=online_factory,
                    tuning_requested=True,
                    deadline=deadline,
                    calibrate=False,
                )
        performance = None
        if (
            candidate_online_result is not None
            and candidate_online_result.online_tuning_allowed
            and candidate_online_result.release_history is not None
        ):
            statistics = candidate_online_result.release_history.rounds[-1]
            performance = OnlinePerformance(
                statistics.median_us,
                statistics.coefficient_of_variation,
            )
        return CandidateAssessment(
            report=outcome.report,
            artifact=outcome.artifact,
            simulation_time_us=(
                None
                if outcome.simulation is None
                else outcome.simulation.completion_time_us
            ),
            online_performance=performance,
            outcome=outcome,
        )

    tuning_context = replace(tuning_context, assess=assessment)
    if online_result is not None:
        tuning_context = attach_online_result_to_tuning_context(
            tuning_context,
            initial.candidate_id,
            online_result,
        )
        tuning_context = replace(tuning_context, assess=assessment)
    result = tune(initial, tuning_context)
    records = []
    for entry in result.history[1:]:
        outcome = entry.outcome or validate_and_lower_candidate(
            entry.schedule,
            inputs,
            topology,
        )
        records.append(
            _CandidateRecord(
                candidate=_tuned_candidate(initial, entry),
                schedule=entry.schedule,
                outcome=outcome,
                topology=topology,
                overlay=entry.overlay,
                tuning_strategy=dict(entry.tuning_strategy),
                accepted=entry.accepted and _offline_valid(outcome),
                rejection_reason=(
                    entry.rejection_reason or _rejection_reason(outcome)
                ),
            )
        )
    return result, tuple(records)


def _finalize(
    *,
    mode: str,
    layout: RunLayout,
    inputs,
    artifacts: Tuple[CandidateArtifact, ...],
    final_candidate_id: Optional[str],
    status: str,
    message: str,
    started: float,
) -> RunArtifacts:
    final_xml = None
    final_report = None
    if final_candidate_id is not None:
        selected = next(
            item for item in artifacts if item.candidate_id == final_candidate_id
        )
        final_xml, final_report = write_final_alias(layout, selected)
    elapsed = max(0.0, _monotonic() - started)
    summary = build_run_summary(
        mode=mode,
        layout=layout,
        inputs=inputs,
        candidates=artifacts,
        final_candidate_id=final_candidate_id,
        final_xml=final_xml,
        final_report=final_report,
        status=status,
        message=message,
        elapsed_s=elapsed,
    )
    write_run_summary(layout, summary)
    return RunArtifacts(
        layout=layout,
        candidates=artifacts,
        final_candidate_id=final_candidate_id,
        final_xml=final_xml,
        final_report=final_report,
        status=status,
        message=message,
    )


def execute_solve(context: RunContext) -> RunArtifacts:
    if not isinstance(context, RunContext):
        raise SemanticError("execute_solve requires a RunContext")
    if context.xml_path is not None:
        raise SemanticError("solve workflow does not accept an XML input")
    if context.online and context.online_context_factory is None:
        raise SemanticError(
            "online validation requires runtime configuration"
        )
    started = _monotonic()
    inputs = resolve_inputs(
        context.topology_path,
        context.sketch_path,
        context.atom_path,
    )
    deadline = _Deadline(
        _timeout(context, inputs.solver.total_solve_timeout_s),
        started,
    )
    layout = create_run_layout(context.output_base, inputs, context.run_id)
    write_resolved_input(layout, inputs)
    deadline.check("input resolution")
    topology = load_topology(inputs)
    plan = build_plan(inputs, topology)
    request = SolveRequest(
        inputs=inputs,
        topology=topology,
        plan=plan,
        solver_version=context.solver_version,
        model_version=context.model_version,
        environment_signature=context.environment_signature,
        wall_clock_budget_s=deadline.remaining(),
    )
    result = solve(request)
    deadline.check("solve")
    schedules = tuple(
        _global_schedule(plan, candidate) for candidate in result.candidates
    )
    outcomes = tuple(
        validate_and_lower_candidate(schedule, inputs, topology)
        for schedule in schedules
    )
    deadline.check("validation")
    final_candidate_id = _select_final(
        result.candidates,
        outcomes,
        result.selected_candidate_id,
    )
    online_result = None
    calibration_message = None
    lineage_records = []
    if context.online and final_candidate_id is not None:
        online_index = next(
            index
            for index, candidate in enumerate(result.candidates)
            if candidate.candidate_id == final_candidate_id
        )
        updated_outcome, online_result = _run_online_candidate(
            candidate_id=final_candidate_id,
            schedule=schedules[online_index],
            outcome=outcomes[online_index],
            inputs=inputs,
            layout=layout,
            factory=context.online_context_factory,
            tuning_requested=context.tune,
            deadline=deadline,
        )
        outcomes = tuple(
            updated_outcome if index == online_index else value
            for index, value in enumerate(outcomes)
        )
        if online_result is not None and online_result.requires_resolve:
            calibration_result = online_result
            calibration_parent_candidate_id = final_candidate_id
            lineage_records.extend(
                _CandidateRecord(
                    candidate=candidate,
                    schedule=schedule,
                    outcome=outcome,
                    topology=topology,
                    overlay=None,
                    tuning_strategy={"kind": "initial_solve"},
                    accepted=_offline_valid(outcome),
                    rejection_reason=_rejection_reason(outcome),
                )
                for candidate, schedule, outcome in zip(
                    result.candidates,
                    schedules,
                    outcomes,
                )
            )
            topology = apply_calibration_to_topology(
                topology,
                _applied_calibration(calibration_result),
            )
            plan = build_plan(inputs, topology)
            request = SolveRequest(
                inputs=inputs,
                topology=topology,
                plan=plan,
                solver_version=context.solver_version,
                model_version=context.model_version,
                environment_signature=context.environment_signature,
                wall_clock_budget_s=deadline.remaining(),
            )
            result = solve(request)
            result = replace(
                result,
                candidates=tuple(
                    replace(
                        candidate,
                        parent_candidate_id=calibration_parent_candidate_id,
                    )
                    for candidate in result.candidates
                ),
            )
            deadline.check("calibrated re-solve")
            schedules = tuple(
                _global_schedule(plan, candidate)
                for candidate in result.candidates
            )
            outcomes = tuple(
                validate_and_lower_candidate(schedule, inputs, topology)
                for schedule in schedules
            )
            deadline.check("calibrated validation")
            final_candidate_id = _select_final(
                result.candidates,
                outcomes,
                result.selected_candidate_id,
            )
            online_result = None
            if final_candidate_id is not None:
                online_index = next(
                    index
                    for index, candidate in enumerate(result.candidates)
                    if candidate.candidate_id == final_candidate_id
                )
                updated_outcome, operator_result = _run_online_candidate(
                    candidate_id=final_candidate_id,
                    schedule=schedules[online_index],
                    outcome=outcomes[online_index],
                    inputs=inputs,
                    layout=layout,
                    factory=context.online_context_factory,
                    tuning_requested=context.tune,
                    deadline=deadline,
                    calibrate=False,
                )
                if operator_result is None:
                    online_result = calibration_result
                else:
                    online_result = _merge_applied_calibration(
                        calibration_result,
                        operator_result,
                    )
                    updated_outcome = _with_online_result(
                        updated_outcome,
                        online_result,
                    )
                outcomes = tuple(
                    updated_outcome if index == online_index else value
                    for index, value in enumerate(outcomes)
                )
            calibration_message = "online_calibration_applied"
    hierarchy = _hierarchy_plan(plan)
    applied = _applied_strategies(inputs)
    records = lineage_records + [
        _CandidateRecord(
            candidate=candidate,
            schedule=schedule,
            outcome=outcome,
            topology=topology,
            overlay=None,
            tuning_strategy={
                "kind": (
                    "calibrated_resolve"
                    if calibration_message is not None
                    else "initial_solve"
                )
            },
            accepted=_offline_valid(outcome),
            rejection_reason=_rejection_reason(outcome),
        )
        for candidate, schedule, outcome in zip(
            result.candidates,
            schedules,
            outcomes,
        )
    ]
    tuning_message = None
    if (
        context.tune
        and final_candidate_id is not None
        and (
            online_result is None or online_result.online_tuning_allowed
        )
    ):
        initial_index = next(
            index
            for index, candidate in enumerate(result.candidates)
            if candidate.candidate_id == final_candidate_id
        )
        tuning_result, tuned = _tuning_records(
            result.candidates[initial_index],
            schedules[initial_index],
            outcomes[initial_index],
            inputs,
            topology,
            deadline,
            online_result=online_result,
            online_factory=context.online_context_factory,
            layout=layout,
        )
        records.extend(tuned)
        if tuning_result.selected_candidate_id is not None:
            final_candidate_id = tuning_result.selected_candidate_id
        tuning_message = tuning_result.stop_reason
    elif context.tune and online_result is not None:
        tuning_message = "online_tuning_not_allowed"
    artifacts = tuple(
        write_candidate_artifact(
            layout,
            inputs,
            record.topology,
            record.candidate,
            record.schedule,
            record.outcome,
            iteration=index,
            selected_best=record.candidate.candidate_id == final_candidate_id,
            accepted=record.accepted,
            rejection_reason=record.rejection_reason,
            applied_strategies=applied,
            hierarchy_plan=hierarchy,
            tuning_strategy=record.tuning_strategy,
            overlay=record.overlay,
        )
        for index, record in enumerate(records)
    )
    deadline.check("artifact writing")
    return _finalize(
        mode="solve",
        layout=layout,
        inputs=inputs,
        artifacts=artifacts,
        final_candidate_id=final_candidate_id,
        status=result.status.value,
        message=(
            (
                result.message
                if calibration_message is None
                else "{}; calibration={}".format(
                    result.message,
                    calibration_message,
                )
            )
            if tuning_message is None
            else "{}; tuning={}".format(
                (
                    result.message
                    if calibration_message is None
                    else "{}; calibration={}".format(
                        result.message,
                        calibration_message,
                    )
                ),
                tuning_message,
            )
        ),
        started=started,
    )


def _verify_source_artifact(
    xml_path: Path,
    sidecar,
    generated: VerificationOutcome,
    inputs,
    topology,
) -> VerificationOutcome:
    if generated.artifact is None:
        return generated
    try:
        xml_text = xml_path.read_text(encoding="utf-8")
    except OSError as error:
        raise SemanticError("verification XML could not be read") from error
    xml_sha256 = hashlib.sha256(xml_text.encode("utf-8")).hexdigest()
    if sidecar.xml_sha256 != xml_sha256:
        raise SemanticError("verification XML hash differs from schedule sidecar")
    reference = generated.artifact
    artifact = XmlArtifact(
        xml_text=xml_text,
        buffer_plan=reference.buffer_plan,
        endpoint_program=reference.endpoint_program,
        tb_program=reference.tb_program,
        sha256=xml_sha256,
        runtime_compatible=reference.runtime_compatible,
    )
    return verify_candidate_outcome(
        sidecar.schedule,
        artifact,
        inputs,
        topology,
    )


def execute_verify(context: RunContext) -> RunArtifacts:
    if not isinstance(context, RunContext):
        raise SemanticError("execute_verify requires a RunContext")
    if context.xml_path is None or context.sidecar_path is None:
        raise SemanticError("verify workflow requires XML and schedule sidecar")
    if context.online and context.online_context_factory is None:
        raise SemanticError(
            "online validation requires runtime configuration"
        )
    started = _monotonic()
    inputs = resolve_inputs(
        context.topology_path,
        context.sketch_path,
        context.atom_path,
    )
    deadline = _Deadline(
        _timeout(context, inputs.hyperparameters.total_verification_timeout_s),
        started,
    )
    layout = create_run_layout(context.output_base, inputs, context.run_id)
    write_resolved_input(layout, inputs)
    sidecar = read_schedule_sidecar(context.sidecar_path)
    if sidecar.normalized_input_sha256 != inputs.input_sha256:
        raise SemanticError("schedule sidecar resolved input hash does not match")
    topology = load_topology(inputs)
    signature = candidate_signature(
        sidecar.schedule,
        inputs,
        topology,
        sidecar.overlay,
    )
    if signature != sidecar.candidate_signature:
        raise SemanticError("schedule sidecar candidate signature does not match")
    deadline.check("sidecar reconstruction")
    generated = validate_and_lower_candidate(
        sidecar.schedule,
        inputs,
        topology,
    )
    outcome = _verify_source_artifact(
        context.xml_path,
        sidecar,
        generated,
        inputs,
        topology,
    )
    deadline.check("verification")
    accepted = _offline_valid(outcome)
    online_result = None
    if context.online and accepted:
        outcome, online_result = _run_online_candidate(
            candidate_id=sidecar.candidate.candidate_id,
            schedule=sidecar.schedule,
            outcome=outcome,
            inputs=inputs,
            layout=layout,
            factory=context.online_context_factory,
            tuning_requested=context.tune,
            deadline=deadline,
        )
        if online_result is not None and online_result.requires_resolve:
            calibration_result = online_result
            outcome, operator_result = _run_online_candidate(
                candidate_id=sidecar.candidate.candidate_id,
                schedule=sidecar.schedule,
                outcome=outcome,
                inputs=inputs,
                layout=layout,
                factory=context.online_context_factory,
                tuning_requested=False,
                deadline=deadline,
                calibrate=False,
            )
            if operator_result is None:
                online_result = calibration_result
            else:
                online_result = _merge_applied_calibration(
                    calibration_result,
                    operator_result,
                    requires_resolve=True,
                )
                outcome = _with_online_result(outcome, online_result)
    records = [
        _CandidateRecord(
            candidate=sidecar.candidate,
            schedule=sidecar.schedule,
            outcome=outcome,
            topology=topology,
            overlay=sidecar.overlay,
            tuning_strategy={"kind": "verify_existing"},
            accepted=accepted,
            rejection_reason=_rejection_reason(outcome),
        )
    ]
    final_candidate_id = sidecar.candidate.candidate_id if accepted else None
    tuning_message = None
    if (
        context.tune
        and accepted
        and (
            online_result is None or online_result.online_tuning_allowed
        )
    ):
        tuning_result, tuned = _tuning_records(
            sidecar.candidate,
            sidecar.schedule,
            outcome,
            inputs,
            topology,
            deadline,
            online_result=online_result,
            online_factory=context.online_context_factory,
            layout=layout,
        )
        records.extend(tuned)
        if tuning_result.selected_candidate_id is not None:
            final_candidate_id = tuning_result.selected_candidate_id
        tuning_message = tuning_result.stop_reason
    elif context.tune and online_result is not None:
        tuning_message = "online_tuning_not_allowed"
    artifacts = tuple(
        write_candidate_artifact(
            layout,
            inputs,
            record.topology,
            record.candidate,
            record.schedule,
            record.outcome,
            iteration=index,
            selected_best=record.candidate.candidate_id
            == final_candidate_id,
            accepted=record.accepted,
            rejection_reason=record.rejection_reason,
            applied_strategies=_applied_strategies(inputs),
            hierarchy_plan={"source": "schedule_sidecar"},
            tuning_strategy=record.tuning_strategy,
            overlay=record.overlay,
        )
        for index, record in enumerate(records)
    )
    deadline.check("artifact writing")
    return _finalize(
        mode="verify",
        layout=layout,
        inputs=inputs,
        artifacts=artifacts,
        final_candidate_id=final_candidate_id,
        status=("valid" if accepted else "invalid"),
        message=(
            (
                "verification complete"
                if tuning_message is None
                else "verification complete; tuning={}".format(
                    tuning_message
                )
            )
            if accepted
            else "verification produced no semantic-valid candidate"
        ),
        started=started,
    )
