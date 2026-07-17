from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import Optional, Tuple

from vericcl.artifacts.hashing import candidate_signature
from vericcl.artifacts.layout import RunLayout, create_run_layout
from vericcl.artifacts.summary import build_run_summary
from vericcl.artifacts.writer import (
    CandidateArtifact,
    read_schedule_sidecar,
    write_candidate_artifact,
    write_final_alias,
    write_resolved_input,
    write_run_summary,
)
from vericcl.composer import compose
from vericcl.errors import SemanticError
from vericcl.input.loader import resolve_inputs
from vericcl.planner.build import build_plan
from vericcl.semantics.atom import Schedule
from vericcl.solver.model import SolveCandidate, SolveRequest
from vericcl.solver.orchestrator import solve
from vericcl.topology.loader import load_topology
from vericcl.verification.model import ValidationStatus
from vericcl.verification.pipeline import (
    VerificationOutcome,
    validate_and_lower_candidate,
    verify_candidate_outcome,
)
from vericcl.xml.lower import XmlArtifact


_monotonic = time.monotonic


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
    if context.online or context.tune:
        raise SemanticError(
            "online validation and tuning require runtime configuration"
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
    hierarchy = _hierarchy_plan(plan)
    applied = _applied_strategies(inputs)
    artifacts = tuple(
        write_candidate_artifact(
            layout,
            inputs,
            topology,
            candidate,
            schedule,
            outcome,
            iteration=index,
            selected_best=candidate.candidate_id == final_candidate_id,
            accepted=_offline_valid(outcome),
            rejection_reason=_rejection_reason(outcome),
            applied_strategies=applied,
            hierarchy_plan=hierarchy,
            tuning_strategy={"kind": "initial_solve"},
        )
        for index, (candidate, schedule, outcome) in enumerate(
            zip(result.candidates, schedules, outcomes)
        )
    )
    deadline.check("artifact writing")
    return _finalize(
        mode="solve",
        layout=layout,
        inputs=inputs,
        artifacts=artifacts,
        final_candidate_id=final_candidate_id,
        status=result.status.value,
        message=result.message,
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
    if context.online or context.tune:
        raise SemanticError(
            "online validation and tuning require runtime configuration"
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
    signature = candidate_signature(sidecar.schedule, inputs, topology, None)
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
    artifact = write_candidate_artifact(
        layout,
        inputs,
        topology,
        sidecar.candidate,
        sidecar.schedule,
        outcome,
        iteration=0,
        selected_best=accepted,
        accepted=accepted,
        rejection_reason=_rejection_reason(outcome),
        applied_strategies=_applied_strategies(inputs),
        hierarchy_plan={"source": "schedule_sidecar"},
        tuning_strategy={"kind": "verify_existing"},
    )
    deadline.check("artifact writing")
    final_candidate_id = sidecar.candidate.candidate_id if accepted else None
    return _finalize(
        mode="verify",
        layout=layout,
        inputs=inputs,
        artifacts=(artifact,),
        final_candidate_id=final_candidate_id,
        status=("valid" if accepted else "invalid"),
        message=(
            "verification complete"
            if accepted
            else "verification produced no semantic-valid candidate"
        ),
        started=started,
    )
