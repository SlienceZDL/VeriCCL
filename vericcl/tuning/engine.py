from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Callable, Mapping, Optional, Tuple

from vericcl.artifacts.hashing import candidate_signature
from vericcl.composer import compose
from vericcl.errors import SemanticError
from vericcl.input.models import ResolvedInput
from vericcl.planner.model import PlanDAG
from vericcl.semantics.atom import Schedule
from vericcl.solver.budget import ModelBudget
from vericcl.solver.model import SolveCandidate
from vericcl.topology.model import Topology
from vericcl.tuning.impact import compute_impact_closure
from vericcl.tuning.local_milp import solve_local_repair
from vericcl.tuning.model import RepairStatus, TuningOverlay
from vericcl.tuning.repair import _reschedule, repair_flow_suffix
from vericcl.verification.bdd_order import TBOrderHint
from vericcl.verification.model import ValidationReport, ValidationStatus
from vericcl.verification.pipeline import (
    VerificationOutcome,
    validate_and_lower_candidate,
)
from vericcl.verification.simulator import simulate_schedule
from vericcl.xml.lower import XmlArtifact
from vericcl.xml.threadblocks import ThreadblockProgram


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SemanticError("{} must be a non-empty string".format(field))
    return value


def _number(value: object, field: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticError("{} must be a number".format(field))
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise SemanticError(
            "{} must be finite and at least {}".format(field, minimum)
        )
    return result


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SemanticError("{} must be a mapping".format(field))
    if not all(isinstance(key, str) and key for key in value):
        raise SemanticError("{} keys must be strings".format(field))
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class OnlinePerformance:
    median_time_us: float
    coefficient_of_variation: float

    def __post_init__(self) -> None:
        median = _number(
            self.median_time_us,
            "online_performance.median_time_us",
            minimum=1e-300,
        )
        variation = _number(
            self.coefficient_of_variation,
            "online_performance.coefficient_of_variation",
        )
        object.__setattr__(self, "median_time_us", median)
        object.__setattr__(self, "coefficient_of_variation", variation)


@dataclass(frozen=True)
class CandidateProposal:
    candidate_id: str
    schedule: Schedule
    overlay: Optional[TuningOverlay]
    parent_candidate_id: Optional[str]
    tuning_strategy: Mapping[str, object]

    def __post_init__(self) -> None:
        _identifier(self.candidate_id, "candidate_proposal.candidate_id")
        if not isinstance(self.schedule, Schedule):
            raise SemanticError("candidate proposal schedule is invalid")
        if self.overlay is not None and not isinstance(
            self.overlay,
            TuningOverlay,
        ):
            raise SemanticError("candidate proposal overlay is invalid")
        if self.parent_candidate_id is not None:
            _identifier(
                self.parent_candidate_id,
                "candidate_proposal.parent_candidate_id",
            )
        if (
            self.overlay is not None
            and self.overlay.parent_candidate_id
            != self.parent_candidate_id
        ):
            raise SemanticError(
                "candidate proposal and overlay parents differ"
            )
        object.__setattr__(
            self,
            "tuning_strategy",
            _mapping(
                self.tuning_strategy,
                "candidate_proposal.tuning_strategy",
            ),
        )


@dataclass(frozen=True)
class CandidateAssessment:
    report: ValidationReport
    artifact: Optional[XmlArtifact]
    simulation_time_us: Optional[float]
    online_performance: Optional[OnlinePerformance]
    outcome: Optional[VerificationOutcome] = None

    def __post_init__(self) -> None:
        if not isinstance(self.report, ValidationReport):
            raise SemanticError("candidate assessment report is invalid")
        if self.artifact is not None and not isinstance(
            self.artifact,
            XmlArtifact,
        ):
            raise SemanticError("candidate assessment artifact is invalid")
        if self.simulation_time_us is not None:
            object.__setattr__(
                self,
                "simulation_time_us",
                _number(
                    self.simulation_time_us,
                    "candidate_assessment.simulation_time_us",
                ),
            )
        if self.online_performance is not None and not isinstance(
            self.online_performance,
            OnlinePerformance,
        ):
            raise SemanticError(
                "candidate assessment online performance is invalid"
            )
        if self.outcome is not None and not isinstance(
            self.outcome,
            VerificationOutcome,
        ):
            raise SemanticError("candidate assessment outcome is invalid")


@dataclass(frozen=True)
class TuningHistoryEntry:
    candidate_id: str
    parent_candidate_id: Optional[str]
    schedule: Schedule
    overlay: Optional[TuningOverlay]
    tuning_strategy: Mapping[str, object]
    candidate_signature: str
    report: Optional[ValidationReport]
    artifact: Optional[XmlArtifact]
    simulation_time_us: Optional[float]
    online_performance: Optional[OnlinePerformance]
    accepted: bool
    rejection_reason: Optional[str]
    selected_best: bool
    offline_analysis_only: bool
    actual_improvement: Optional[float]
    required_improvement: Optional[float]
    outcome: Optional[VerificationOutcome]

    def __post_init__(self) -> None:
        _identifier(self.candidate_id, "tuning_history.candidate_id")
        _identifier(
            self.candidate_signature,
            "tuning_history.candidate_signature",
        )
        object.__setattr__(
            self,
            "tuning_strategy",
            _mapping(self.tuning_strategy, "tuning_history.tuning_strategy"),
        )


@dataclass(frozen=True)
class TuningResult:
    selected_candidate_id: Optional[str]
    selected_schedule: Optional[Schedule]
    selected_artifact: Optional[XmlArtifact]
    history: Tuple[TuningHistoryEntry, ...]
    stop_reason: str
    iterations: int

    def __post_init__(self) -> None:
        history = tuple(self.history)
        if not all(isinstance(item, TuningHistoryEntry) for item in history):
            raise SemanticError("tuning result history is invalid")
        ids = tuple(item.candidate_id for item in history)
        if len(ids) != len(set(ids)):
            raise SemanticError("tuning result candidate IDs must be unique")
        object.__setattr__(self, "history", history)
        if self.selected_candidate_id is not None:
            _identifier(
                self.selected_candidate_id,
                "tuning_result.selected_candidate_id",
            )
            if self.selected_candidate_id not in ids:
                raise SemanticError("selected tuning candidate is missing")
        _identifier(self.stop_reason, "tuning_result.stop_reason")


AssessFunction = Callable[[CandidateProposal], CandidateAssessment]
GenerateFunction = Callable[
    [TuningHistoryEntry, int],
    Tuple[CandidateProposal, ...],
]
SimulateFunction = Callable[[Schedule], float]


@dataclass(frozen=True)
class TuningContext:
    inputs: ResolvedInput
    topology: Topology
    initial_schedule: Optional[Schedule] = None
    plan: Optional[PlanDAG] = None
    assess: Optional[AssessFunction] = None
    generate: Optional[GenerateFunction] = None
    simulate: Optional[SimulateFunction] = None
    online_validation: bool = False
    online_performance: Mapping[str, OnlinePerformance] = MappingProxyType({})
    max_iterations: Optional[int] = None
    timeout_s: Optional[float] = None
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        if not isinstance(self.inputs, ResolvedInput):
            raise SemanticError("tuning context inputs are invalid")
        if not isinstance(self.topology, Topology):
            raise SemanticError("tuning context topology is invalid")
        if self.initial_schedule is not None and not isinstance(
            self.initial_schedule,
            Schedule,
        ):
            raise SemanticError("tuning context initial schedule is invalid")
        if self.plan is not None and not isinstance(self.plan, PlanDAG):
            raise SemanticError("tuning context plan is invalid")
        if self.initial_schedule is None and self.plan is None:
            raise SemanticError(
                "tuning context requires an initial schedule or plan"
            )
        for function, field in (
            (self.assess, "assess"),
            (self.generate, "generate"),
            (self.simulate, "simulate"),
        ):
            if function is not None and not callable(function):
                raise SemanticError(
                    "tuning context {} must be callable".format(field)
                )
        if not isinstance(self.online_validation, bool):
            raise SemanticError("online_validation must be a boolean")
        online = dict(self.online_performance)
        if not all(
            isinstance(key, str)
            and key
            and isinstance(value, OnlinePerformance)
            for key, value in online.items()
        ):
            raise SemanticError("tuning context online performance is invalid")
        object.__setattr__(
            self,
            "online_performance",
            MappingProxyType(online),
        )
        if self.max_iterations is not None and (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or self.max_iterations < 1
        ):
            raise SemanticError("max_iterations must be a positive integer")
        if self.timeout_s is not None:
            object.__setattr__(
                self,
                "timeout_s",
                _number(
                    self.timeout_s,
                    "tuning_context.timeout_s",
                    minimum=1e-300,
                ),
            )
        if not callable(self.clock):
            raise SemanticError("tuning context clock must be callable")


def _initial_schedule(
    initial: SolveCandidate,
    context: TuningContext,
) -> Schedule:
    if context.initial_schedule is not None:
        return context.initial_schedule
    assert context.plan is not None
    return compose(
        context.plan,
        {node.node_id: initial for node in context.plan.nodes},
    )


def _default_assess(
    proposal: CandidateProposal,
    context: TuningContext,
) -> CandidateAssessment:
    outcome = validate_and_lower_candidate(
        proposal.schedule,
        context.inputs,
        context.topology,
    )
    completion = (
        outcome.simulation.completion_time_us
        if outcome.simulation is not None
        else None
    )
    return CandidateAssessment(
        outcome.report,
        outcome.artifact,
        completion,
        context.online_performance.get(proposal.candidate_id),
        outcome,
    )


def _rejection(
    assessment: CandidateAssessment,
    online_validation: bool,
) -> Tuple[Optional[str], bool]:
    report = assessment.report
    if report.overall_status is not ValidationStatus.VALID:
        return "correctness_invalid", False
    if report.bdd.status is ValidationStatus.ANALYSIS_ERROR:
        return "bdd_analysis_error", False
    if report.bdd.status is not ValidationStatus.VALID:
        return "bdd_analysis_incomplete", False
    if (
        report.simulation.status is not ValidationStatus.VALID
        or assessment.simulation_time_us is None
    ):
        return "simulation_incomplete", False
    if not report.runtime_compatible:
        return "runtime_incompatible", True
    if online_validation:
        if report.online.status is not ValidationStatus.VALID:
            return "online_validation_incomplete", False
        if assessment.online_performance is None:
            return "online_performance_missing", False
    return None, False


def _entry(
    proposal: CandidateProposal,
    signature: str,
    assessment: Optional[CandidateAssessment],
    *,
    accepted: bool,
    rejection_reason: Optional[str],
    offline_analysis_only: bool = False,
    actual_improvement: Optional[float] = None,
    required_improvement: Optional[float] = None,
) -> TuningHistoryEntry:
    return TuningHistoryEntry(
        candidate_id=proposal.candidate_id,
        parent_candidate_id=proposal.parent_candidate_id,
        schedule=proposal.schedule,
        overlay=proposal.overlay,
        tuning_strategy=proposal.tuning_strategy,
        candidate_signature=signature,
        report=assessment.report if assessment is not None else None,
        artifact=assessment.artifact if assessment is not None else None,
        simulation_time_us=(
            assessment.simulation_time_us
            if assessment is not None
            else None
        ),
        online_performance=(
            assessment.online_performance
            if assessment is not None
            else None
        ),
        accepted=accepted,
        rejection_reason=rejection_reason,
        selected_best=False,
        offline_analysis_only=offline_analysis_only,
        actual_improvement=actual_improvement,
        required_improvement=required_improvement,
        outcome=assessment.outcome if assessment is not None else None,
    )


def _timed_transfer(transfer, start: float):
    duration = transfer.ed_time - transfer.st_time
    end = start + duration
    return replace(
        transfer,
        atoms=tuple(
            replace(atom, st_time=start, ed_time=end)
            for atom in transfer.atoms
        ),
        st_time=start,
        ed_time=end,
    )


def _repair_tb_order(
    schedule: Schedule,
    program: ThreadblockProgram,
    hint: TBOrderHint,
    overlay_id: str,
) -> Schedule:
    if not isinstance(schedule, Schedule):
        raise SemanticError("schedule must be a Schedule")
    if not isinstance(program, ThreadblockProgram):
        raise SemanticError("program must be a ThreadblockProgram")
    if not isinstance(hint, TBOrderHint):
        raise SemanticError("hint must be a TBOrderHint")
    _identifier(overlay_id, "order_overlay_id")
    try:
        earlier_step = program.steps_by_id[hint.earlier_step_id]
        later_step = program.steps_by_id[hint.later_step_id]
    except KeyError as error:
        raise SemanticError("TB order hint references a missing step") from error
    earlier_id = earlier_step.transfer_id
    later_id = later_step.transfer_id
    if earlier_id == later_id:
        raise SemanticError("TB order hint references one transfer twice")
    by_id = {
        transfer.transfer_id: transfer for transfer in schedule.transfers
    }
    if earlier_id not in by_id or later_id not in by_id:
        raise SemanticError("TB order hint transfer is missing")
    earlier = by_id[earlier_id]
    later = by_id[later_id]
    if (
        earlier.src_rank,
        earlier.dst_rank,
        earlier.channel,
    ) != (
        later.src_rank,
        later.dst_rank,
        later.channel,
    ):
        raise SemanticError("TB order hint transfers do not share a lane")
    raw_semantic = schedule.metadata.get("semantic_predecessors", {})
    if not isinstance(raw_semantic, Mapping):
        raise SemanticError("semantic_predecessors must be a mapping")
    semantic = {
        transfer.transfer_id: frozenset(
            raw_semantic.get(
                transfer.transfer_id,
                transfer.predecessor_ids,
            )
        )
        for transfer in schedule.transfers
    }
    if (
        earlier_id in semantic[later_id]
        or later_id in semantic[earlier_id]
    ):
        raise SemanticError("TB order hint would reverse semantic precedence")

    base_start = min(earlier.st_time, later.st_time)
    later = _timed_transfer(later, base_start)
    later = replace(
        later,
        predecessor_ids=later.predecessor_ids - {earlier_id},
    )
    earlier_ready = max(
        atom.current_symbol.ready_time for atom in earlier.atoms
    )
    earlier = _timed_transfer(
        earlier,
        max(earlier_ready, later.ed_time),
    )
    earlier = replace(
        earlier,
        predecessor_ids=earlier.predecessor_ids | {later_id},
    )
    transfers = tuple(
        (
            earlier
            if transfer.transfer_id == earlier_id
            else later
            if transfer.transfer_id == later_id
            else transfer
        )
        for transfer in schedule.transfers
    )
    raw_slots = schedule.metadata.get("resource_slots", {})
    if not isinstance(raw_slots, Mapping):
        raise SemanticError("resource_slots must be a mapping")
    resource_slots = {
        transfer.transfer_id: dict(
            raw_slots.get(transfer.transfer_id, {})
        )
        for transfer in transfers
    }
    transfers = _reschedule(transfers, semantic, resource_slots)
    return replace(
        schedule,
        schedule_id="{}-{}".format(schedule.schedule_id, overlay_id),
        transfers=transfers,
    )


def _builtin_generate(
    current: TuningHistoryEntry,
    iteration: int,
    context: TuningContext,
    remaining_seconds: float,
) -> Tuple[CandidateProposal, ...]:
    outcome = current.outcome
    if outcome is None or outcome.flow_bdd is None:
        return ()
    proposals = []
    for hint_index, hint in enumerate(outcome.flow_bdd.hints):
        overlay = TuningOverlay(
            overlay_id="tune-i{:02d}-h{:04d}".format(
                iteration,
                hint_index,
            ),
            parent_candidate_id=current.candidate_id,
        )
        repair = repair_flow_suffix(
            current.schedule,
            hint,
            overlay,
            context.topology,
            context.inputs,
        )
        method = "greedy"
        if (
            repair.status is not RepairStatus.SUCCESS
            and hint.earliest_candidate_start_us < hint.wait_end_us
            and remaining_seconds > 0.0
        ):
            impact = compute_impact_closure(
                current.schedule,
                frozenset({hint.waiting_transfer_id}),
                context.topology,
            )
            repair = solve_local_repair(
                current.schedule,
                hint,
                impact,
                overlay,
                context.topology,
                context.inputs,
                ModelBudget(
                    remaining_seconds,
                    0.0,
                    remaining_seconds,
                ),
            )
            method = "local_milp"
        if repair.status is not RepairStatus.SUCCESS:
            continue
        proposals.append(
            CandidateProposal(
                candidate_id="{}-{}".format(
                    overlay.overlay_id,
                    repair.selected_candidate_flow_id,
                ),
                schedule=repair.schedule,
                overlay=overlay,
                parent_candidate_id=current.candidate_id,
                tuning_strategy={
                    "kind": "flow_suffix",
                    "method": method,
                    "source_flow_id": hint.source_flow_id,
                    "waiting_transfer_id": hint.waiting_transfer_id,
                    "repair_evidence": dict(repair.evidence),
                },
            )
        )
    if outcome.order_bdd is not None and outcome.artifact is not None:
        for hint_index, hint in enumerate(outcome.order_bdd.hints):
            earlier = outcome.artifact.tb_program.steps_by_id[
                hint.earlier_step_id
            ].transfer_id
            later = outcome.artifact.tb_program.steps_by_id[
                hint.later_step_id
            ].transfer_id
            overlay = TuningOverlay(
                overlay_id="tune-order-i{:02d}-h{:04d}".format(
                    iteration,
                    hint_index,
                ),
                parent_candidate_id=current.candidate_id,
                lane_order=((later, earlier),),
            )
            try:
                schedule = _repair_tb_order(
                    current.schedule,
                    outcome.artifact.tb_program,
                    hint,
                    overlay.overlay_id,
                )
            except SemanticError:
                continue
            proposals.append(
                CandidateProposal(
                    candidate_id=overlay.overlay_id,
                    schedule=schedule,
                    overlay=overlay,
                    parent_candidate_id=current.candidate_id,
                    tuning_strategy={
                        "kind": "tb_order",
                        "tb_id": hint.tb_id,
                        "earlier_step_id": hint.earlier_step_id,
                        "later_step_id": hint.later_step_id,
                    },
                )
            )
    return tuple(proposals)


def _comparison(
    best: TuningHistoryEntry,
    assessment: CandidateAssessment,
    context: TuningContext,
) -> Tuple[bool, float, float, str]:
    if context.online_validation:
        assert best.online_performance is not None
        assert assessment.online_performance is not None
        baseline = best.online_performance.median_time_us
        candidate = assessment.online_performance.median_time_us
        improvement = (baseline - candidate) / baseline
        required = max(
            context.inputs.hyperparameters.min_tuning_improvement,
            2.0
            * max(
                best.online_performance.coefficient_of_variation,
                assessment.online_performance.coefficient_of_variation,
            ),
        )
        return (
            improvement >= required,
            improvement,
            required,
            "online_improvement_below_threshold",
        )
    assert best.simulation_time_us is not None
    assert assessment.simulation_time_us is not None
    baseline = best.simulation_time_us
    candidate = assessment.simulation_time_us
    improvement = (
        (baseline - candidate) / baseline
        if baseline > 0.0
        else (1.0 if candidate < baseline else 0.0)
    )
    return candidate < baseline, improvement, 0.0, "no_improvement"


def tune(initial: SolveCandidate, context: TuningContext) -> TuningResult:
    if not isinstance(initial, SolveCandidate):
        raise SemanticError("initial must be a SolveCandidate")
    if not isinstance(context, TuningContext):
        raise SemanticError("context must be a TuningContext")
    schedule = _initial_schedule(initial, context)
    initial_proposal = CandidateProposal(
        candidate_id=initial.candidate_id,
        schedule=schedule,
        overlay=None,
        parent_candidate_id=initial.parent_candidate_id,
        tuning_strategy={"kind": "initial"},
    )
    timeout = (
        context.timeout_s
        if context.timeout_s is not None
        else float(
            context.inputs.hyperparameters.total_verification_timeout_s
        )
    )
    started_at = context.clock()
    assess = context.assess or (
        lambda proposal: _default_assess(proposal, context)
    )
    initial_assessment = assess(initial_proposal)
    signature = candidate_signature(
        schedule,
        context.inputs,
        context.topology,
        None,
    )
    initial_timed_out = context.clock() - started_at >= timeout
    if initial_timed_out:
        reason, offline_only = "verification_timeout", False
    else:
        reason, offline_only = _rejection(
            initial_assessment,
            context.online_validation,
        )
    initial_entry = _entry(
        initial_proposal,
        signature,
        initial_assessment,
        accepted=reason is None,
        rejection_reason=reason,
        offline_analysis_only=offline_only,
    )
    history = [initial_entry]
    best = initial_entry if initial_entry.accepted else None
    signatures = {signature}
    candidate_ids = {initial.candidate_id}
    iteration_limit = min(
        20,
        context.inputs.hyperparameters.max_tuning_iterations,
        (
            context.max_iterations
            if context.max_iterations is not None
            else context.inputs.hyperparameters.max_tuning_iterations
        ),
    )
    stop_reason = "candidate_space_exhausted"
    completed_iterations = 0

    iterations = () if initial_timed_out else range(iteration_limit)
    for iteration in iterations:
        elapsed = context.clock() - started_at
        if elapsed >= timeout:
            stop_reason = "verification_timeout"
            break
        if best is None:
            stop_reason = "no_eligible_initial_candidate"
            break
        remaining = max(0.0, timeout - elapsed)
        generation_parent_id = best.candidate_id
        proposals = (
            context.generate(best, iteration)
            if context.generate is not None
            else _builtin_generate(best, iteration, context, remaining)
        )
        try:
            proposals = tuple(proposals)
        except TypeError as error:
            raise SemanticError(
                "candidate generator must return an iterable"
            ) from error
        if not all(isinstance(item, CandidateProposal) for item in proposals):
            raise SemanticError(
                "candidate generator must return CandidateProposal values"
            )
        if not proposals:
            stop_reason = "candidate_space_exhausted"
            break
        completed_iterations += 1
        iteration_timed_out = False
        for proposal in proposals:
            if proposal.candidate_id in candidate_ids:
                raise SemanticError("tuning candidate IDs must be unique")
            if proposal.parent_candidate_id != generation_parent_id:
                raise SemanticError(
                    "tuning proposal parent does not match generation parent"
                )
            candidate_ids.add(proposal.candidate_id)
            proposal_signature = candidate_signature(
                proposal.schedule,
                context.inputs,
                context.topology,
                proposal.overlay,
            )
            if proposal_signature in signatures:
                history.append(
                    _entry(
                        proposal,
                        proposal_signature,
                        None,
                        accepted=False,
                        rejection_reason="duplicate_candidate_signature",
                    )
                )
                continue
            signatures.add(proposal_signature)
            if proposal.overlay is not None:
                try:
                    proposal.overlay.validate_against(
                        context.inputs,
                        proposal.schedule,
                        context.topology,
                    )
                except SemanticError:
                    history.append(
                        _entry(
                            proposal,
                            proposal_signature,
                            None,
                            accepted=False,
                            rejection_reason="invalid_tuning_overlay",
                        )
                    )
                    continue
            if context.clock() - started_at >= timeout:
                history.append(
                    _entry(
                        proposal,
                        proposal_signature,
                        None,
                        accepted=False,
                        rejection_reason="verification_timeout",
                    )
                )
                iteration_timed_out = True
                continue
            simulation_function = context.simulate or (
                lambda value: simulate_schedule(
                    value,
                    context.topology,
                ).completion_time_us
            )
            try:
                simulated_time = _number(
                    simulation_function(proposal.schedule),
                    "incremental_simulation_time_us",
                )
            except SemanticError:
                history.append(
                    _entry(
                        proposal,
                        proposal_signature,
                        None,
                        accepted=False,
                        rejection_reason="incremental_simulation_failed",
                    )
                )
                continue
            if context.clock() - started_at >= timeout:
                history.append(
                    _entry(
                        proposal,
                        proposal_signature,
                        None,
                        accepted=False,
                        rejection_reason="verification_timeout",
                    )
                )
                iteration_timed_out = True
                continue
            if (
                best.simulation_time_us is not None
                and simulated_time >= best.simulation_time_us
            ):
                baseline = best.simulation_time_us
                improvement = (
                    (baseline - simulated_time) / baseline
                    if baseline > 0.0
                    else 0.0
                )
                history.append(
                    _entry(
                        proposal,
                        proposal_signature,
                        None,
                        accepted=False,
                        rejection_reason="no_simulated_improvement",
                        actual_improvement=improvement,
                        required_improvement=0.0,
                    )
                )
                continue
            assessment = assess(proposal)
            if context.clock() - started_at >= timeout:
                history.append(
                    _entry(
                        proposal,
                        proposal_signature,
                        assessment,
                        accepted=False,
                        rejection_reason="verification_timeout",
                    )
                )
                iteration_timed_out = True
                continue
            reason, offline_only = _rejection(
                assessment,
                context.online_validation,
            )
            if reason is not None:
                history.append(
                    _entry(
                        proposal,
                        proposal_signature,
                        assessment,
                        accepted=False,
                        rejection_reason=reason,
                        offline_analysis_only=offline_only,
                    )
                )
                continue
            assert best is not None
            accepted, improvement, required, rejection = _comparison(
                best,
                assessment,
                context,
            )
            entry = _entry(
                proposal,
                proposal_signature,
                assessment,
                accepted=accepted,
                rejection_reason=None if accepted else rejection,
                actual_improvement=improvement,
                required_improvement=required,
            )
            history.append(entry)
            if accepted:
                best = entry
        if iteration_timed_out:
            stop_reason = "verification_timeout"
            break
    else:
        stop_reason = (
            "verification_timeout"
            if initial_timed_out
            else "max_tuning_iterations"
        )

    selected_id = best.candidate_id if best is not None else None
    history = [
        replace(
            entry,
            selected_best=(entry.candidate_id == selected_id),
        )
        for entry in history
    ]
    selected = next(
        (entry for entry in history if entry.selected_best),
        None,
    )
    return TuningResult(
        selected_candidate_id=selected_id,
        selected_schedule=selected.schedule if selected is not None else None,
        selected_artifact=selected.artifact if selected is not None else None,
        history=tuple(history),
        stop_reason=stop_reason,
        iterations=completed_iterations,
    )
