from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional, Tuple
import uuid

from vericcl.artifacts.hashing import candidate_signature
from vericcl.artifacts.layout import RunLayout
from vericcl.artifacts.reports import (
    build_candidate_report,
    build_validation_json,
)
from vericcl.errors import SemanticError
from vericcl.input.json_codec import canonical_json
from vericcl.input.models import ForbiddenTransfer, ObjectiveMode, ResolvedInput
from vericcl.semantics.atom import Atom, PathStage, Schedule, Symbol, Transfer
from vericcl.solver.model import (
    SolveCandidate,
    SolveStatus,
    SolverMetrics,
)
from vericcl.topology.model import Topology
from vericcl.tuning.model import TuningOverlay
from vericcl.verification.model import ValidationStatus
from vericcl.verification.pipeline import VerificationOutcome


_SIDECAR_SCHEMA_VERSION = 1


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    if not isinstance(path, Path):
        raise SemanticError("atomic write path must be a Path")
    if not isinstance(data, bytes):
        raise SemanticError("atomic write data must be bytes")
    if path.exists():
        raise FileExistsError("artifact already exists: {}".format(path))
    temporary = path.with_name(
        ".{}.{}.tmp".format(path.name, uuid.uuid4().hex)
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError("artifact already exists: {}".format(path))
        os.rename(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(path: Path, text: str) -> None:
    if not isinstance(text, str):
        raise SemanticError("atomic write text must be a string")
    atomic_write_bytes(path, text.encode("utf-8"))


def _validation_statuses(outcome: VerificationOutcome) -> Mapping[str, str]:
    report = outcome.report
    return MappingProxyType(
        {
            field: getattr(report, field).status.value
            for field in (
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
                "runtime",
                "online",
            )
        }
    )


def _offline_valid(outcome: VerificationOutcome) -> bool:
    report = outcome.report
    return (
        report.overall_status is ValidationStatus.VALID
        and report.bdd.status is ValidationStatus.VALID
        and report.simulation.status is ValidationStatus.VALID
    )


@dataclass(frozen=True)
class CandidateArtifact:
    candidate_id: str
    parent_candidate_id: Optional[str]
    iteration: int
    xml_path: Optional[Path]
    report_path: Path
    schedule_path: Path
    xml_sha256: Optional[str]
    report_sha256: str
    candidate_signature: str
    binding_sha256: Optional[str]
    validation: Mapping[str, str]
    runtime_compatible: bool
    accepted: bool
    rejection_reason: Optional[str]
    selected_best: bool
    proven_optimal: bool
    restrictions: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise SemanticError("candidate artifact ID is invalid")
        if isinstance(self.iteration, bool) or not isinstance(self.iteration, int):
            raise SemanticError("candidate artifact iteration must be an integer")
        if self.iteration < 0:
            raise SemanticError("candidate artifact iteration must be non-negative")
        if self.xml_path is not None and not isinstance(self.xml_path, Path):
            raise SemanticError("candidate artifact XML path is invalid")
        if not isinstance(self.report_path, Path) or not isinstance(
            self.schedule_path,
            Path,
        ):
            raise SemanticError("candidate artifact paths are invalid")
        if not isinstance(self.validation, Mapping):
            raise SemanticError("candidate artifact validation is invalid")
        object.__setattr__(
            self,
            "validation",
            MappingProxyType(dict(self.validation)),
        )
        object.__setattr__(self, "restrictions", tuple(self.restrictions))


@dataclass(frozen=True)
class ScheduleSidecar:
    normalized_input_sha256: str
    candidate_signature: str
    xml_sha256: Optional[str]
    candidate: SolveCandidate
    schedule: Schedule
    overlay: Optional[TuningOverlay]


def _resolved_payload(inputs: ResolvedInput) -> Mapping[str, object]:
    return {
        "schema_version": "1",
        "input_sha256": inputs.input_sha256,
        "topology": inputs.resolved_topology,
        "sketch": inputs.resolved_sketch,
        "atom": inputs.resolved_atom,
    }


def write_resolved_input(layout: RunLayout, inputs: ResolvedInput) -> Path:
    if not isinstance(layout, RunLayout):
        raise SemanticError("layout must be a RunLayout")
    if not isinstance(inputs, ResolvedInput):
        raise SemanticError("inputs must be a ResolvedInput")
    atomic_write_text(
        layout.resolved_input,
        canonical_json(_resolved_payload(inputs)) + "\n",
    )
    return layout.resolved_input


def _schedule_payload(schedule: Schedule) -> Mapping[str, object]:
    return {
        "schedule_id": schedule.schedule_id,
        "transfers": schedule.transfers,
        "final_state_ids": schedule.final_state_ids,
        "rank_count": schedule.rank_count,
        "slice_count": schedule.slice_count,
        "slice_size_bytes": schedule.slice_size_bytes,
        "metadata": schedule.metadata,
    }


def _candidate_payload(candidate: SolveCandidate) -> Mapping[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "objective_mode": candidate.objective_mode.value,
        "channel_count": candidate.channel_count,
        "metrics": candidate.metrics,
        "selected_best": candidate.selected_best,
        "proven_optimal": candidate.proven_optimal,
        "search_space_restricted": candidate.search_space_restricted,
        "restrictions": candidate.restrictions,
        "parent_candidate_id": candidate.parent_candidate_id,
    }


def _sidecar_payload(
    inputs: ResolvedInput,
    candidate: SolveCandidate,
    schedule: Schedule,
    signature: str,
    xml_sha256: Optional[str],
    overlay: Optional[TuningOverlay],
) -> Mapping[str, object]:
    return {
        "schema_version": _SIDECAR_SCHEMA_VERSION,
        "normalized_input_sha256": inputs.input_sha256,
        "candidate_signature": signature,
        "xml_sha256": xml_sha256,
        "candidate": _candidate_payload(candidate),
        "overlay": overlay,
        "schedule": _schedule_payload(schedule),
    }


def _parse_symbol(payload: object) -> Symbol:
    if not isinstance(payload, Mapping):
        raise SemanticError("schedule sidecar symbol is invalid")
    return Symbol(
        src_rank=payload.get("src_rank"),
        dst_rank=payload.get("dst_rank"),
        ready_time=payload.get("ready_time"),
    )


def _parse_stage(payload: object) -> PathStage:
    if not isinstance(payload, Mapping):
        raise SemanticError("schedule sidecar path stage is invalid")
    return PathStage(
        stage_id=payload.get("stage_id"),
        operator=payload.get("operator"),
        symbols=tuple(
            _parse_symbol(value) for value in payload.get("symbols", ())
        ),
    )


def _parse_atom(payload: object) -> Atom:
    if not isinstance(payload, Mapping):
        raise SemanticError("schedule sidecar atom is invalid")
    return Atom(
        slice_id=payload.get("slice_id"),
        slice_size_bytes=payload.get("slice_size_bytes"),
        path=tuple(_parse_stage(value) for value in payload.get("path", ())),
        st_time=payload.get("st_time"),
        ed_time=payload.get("ed_time"),
    )


def _parse_transfer(payload: object) -> Transfer:
    if not isinstance(payload, Mapping):
        raise SemanticError("schedule sidecar transfer is invalid")
    return Transfer(
        transfer_id=payload.get("transfer_id"),
        kind=payload.get("kind"),
        src_rank=payload.get("src_rank"),
        dst_rank=payload.get("dst_rank"),
        channel=payload.get("channel"),
        stage_id=payload.get("stage_id"),
        member_slice_ids=frozenset(payload.get("member_slice_ids", ())),
        atoms=tuple(_parse_atom(value) for value in payload.get("atoms", ())),
        st_time=payload.get("st_time"),
        ed_time=payload.get("ed_time"),
        predecessor_ids=frozenset(payload.get("predecessor_ids", ())),
    )


def _parse_schedule(payload: object) -> Schedule:
    if not isinstance(payload, Mapping):
        raise SemanticError("schedule sidecar schedule is invalid")
    return Schedule(
        schedule_id=payload.get("schedule_id"),
        transfers=tuple(
            _parse_transfer(value) for value in payload.get("transfers", ())
        ),
        final_state_ids=tuple(payload.get("final_state_ids", ())),
        rank_count=payload.get("rank_count"),
        slice_count=payload.get("slice_count"),
        slice_size_bytes=payload.get("slice_size_bytes"),
        metadata=payload.get("metadata"),
    )


def _parse_metrics(payload: object) -> SolverMetrics:
    if not isinstance(payload, Mapping):
        raise SemanticError("schedule sidecar solver metrics are invalid")
    try:
        status = SolveStatus(payload.get("status"))
    except ValueError as error:
        raise SemanticError("schedule sidecar solver status is invalid") from error
    return SolverMetrics(
        status=status,
        objective_values=tuple(payload.get("objective_values", ())),
        best_bound=payload.get("best_bound"),
        mip_gap=payload.get("mip_gap"),
        within_requested_gap=payload.get("within_requested_gap"),
        solve_time_s=payload.get("solve_time_s"),
        model_count=payload.get("model_count"),
        operation_count=payload.get("operation_count"),
        hop_count=payload.get("hop_count"),
        makespan_us=payload.get("makespan_us"),
        maximum_normalized_resource_load=payload.get(
            "maximum_normalized_resource_load"
        ),
        solver_name=payload.get("solver_name"),
        solver_version=payload.get("solver_version"),
        solver_seed=payload.get("solver_seed"),
        thread_count=payload.get("thread_count"),
        termination_reason=payload.get("termination_reason"),
        model_index=payload.get("model_index", 0),
    )


def _parse_candidate(payload: object, schedule: Schedule) -> SolveCandidate:
    if not isinstance(payload, Mapping):
        raise SemanticError("schedule sidecar candidate is invalid")
    try:
        objective = ObjectiveMode(payload.get("objective_mode"))
    except ValueError as error:
        raise SemanticError("schedule sidecar objective mode is invalid") from error
    return SolveCandidate(
        candidate_id=payload.get("candidate_id"),
        node_schedules={"global": schedule},
        objective_mode=objective,
        channel_count=payload.get("channel_count"),
        metrics=_parse_metrics(payload.get("metrics")),
        selected_best=payload.get("selected_best"),
        proven_optimal=payload.get("proven_optimal"),
        search_space_restricted=payload.get("search_space_restricted"),
        restrictions=tuple(payload.get("restrictions", ())),
        parent_candidate_id=payload.get("parent_candidate_id"),
    )


def _parse_overlay(payload: object) -> Optional[TuningOverlay]:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise SemanticError("schedule sidecar overlay is invalid")
    forbidden = frozenset(
        ForbiddenTransfer(
            slice_id=value.get("slice_id"),
            src_rank=value.get("src_rank"),
            dst_rank=value.get("dst_rank"),
            stage_id=value.get("stage_id"),
        )
        for value in payload.get("temporary_forbidden", ())
        if isinstance(value, Mapping)
    )
    if len(forbidden) != len(payload.get("temporary_forbidden", ())):
        raise SemanticError("schedule sidecar forbidden overlay is invalid")
    return TuningOverlay(
        overlay_id=payload.get("overlay_id"),
        parent_candidate_id=payload.get("parent_candidate_id"),
        channel_count=payload.get("channel_count"),
        path_weights=tuple(
            tuple(value) for value in payload.get("path_weights", ())
        ),
        temporary_forbidden=forbidden,
        batch_size=payload.get("batch_size"),
        tree_roots=tuple(tuple(value) for value in payload.get("tree_roots", ())),
        tree_edges=tuple(tuple(value) for value in payload.get("tree_edges", ())),
        lane_order=tuple(tuple(value) for value in payload.get("lane_order", ())),
        milp_parameters=tuple(
            tuple(value) for value in payload.get("milp_parameters", ())
        ),
        warm_start_candidate_id=payload.get("warm_start_candidate_id"),
        resolve_scope=tuple(payload.get("resolve_scope", ())),
        hierarchy_template=payload.get("hierarchy_template"),
    )


def read_schedule_sidecar(path: Path) -> ScheduleSidecar:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise SemanticError("schedule sidecar could not be read") from error
    if not isinstance(payload, Mapping):
        raise SemanticError("schedule sidecar root is invalid")
    if payload.get("schema_version") != _SIDECAR_SCHEMA_VERSION:
        raise SemanticError("schedule sidecar schema version is unsupported")
    schedule = _parse_schedule(payload.get("schedule"))
    candidate = _parse_candidate(payload.get("candidate"), schedule)
    overlay = _parse_overlay(payload.get("overlay"))
    return ScheduleSidecar(
        normalized_input_sha256=payload.get("normalized_input_sha256"),
        candidate_signature=payload.get("candidate_signature"),
        xml_sha256=payload.get("xml_sha256"),
        candidate=candidate,
        schedule=schedule,
        overlay=overlay,
    )


def load_schedule_sidecar(path: Path) -> Schedule:
    return read_schedule_sidecar(path).schedule


def _candidate_base(
    layout: RunLayout,
    iteration: int,
    selected_best: bool,
) -> str:
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise SemanticError("iteration must be a non-negative integer")
    if not isinstance(selected_best, bool):
        raise SemanticError("selected_best must be a boolean")
    return "{}_iter-{:03d}_selected-best-{}".format(
        layout.artifact_prefix,
        iteration,
        str(selected_best).lower(),
    )


def _generic_report(
    candidate: SolveCandidate,
    inputs: ResolvedInput,
    outcome: VerificationOutcome,
    signature: str,
) -> Mapping[str, object]:
    return {
        "schema_version": "1",
        "candidate_id": candidate.candidate_id,
        "normalized_input_sha256": inputs.input_sha256,
        "candidate_signature": signature,
        "artifact_binding_sha256": None,
        "solver_metrics": candidate.metrics,
        "validation": outcome.report,
        "lineage": {
            "candidate_id": candidate.candidate_id,
            "parent_candidate_id": candidate.parent_candidate_id,
        },
        "proven_optimal": candidate.proven_optimal,
        "search_space_restricted": candidate.search_space_restricted,
        "restrictions": candidate.restrictions,
        "runtime_compatible": False,
        "xml_sha256": None,
    }


def write_candidate_artifact(
    layout: RunLayout,
    inputs: ResolvedInput,
    topology: Topology,
    candidate: SolveCandidate,
    schedule: Schedule,
    outcome: VerificationOutcome,
    *,
    iteration: int,
    selected_best: bool,
    accepted: bool,
    rejection_reason: Optional[str],
    applied_strategies: Optional[Mapping[str, object]] = None,
    hierarchy_plan: Optional[Mapping[str, object]] = None,
    tuning_strategy: Optional[Mapping[str, object]] = None,
    overlay: Optional[TuningOverlay] = None,
) -> CandidateArtifact:
    if not isinstance(layout, RunLayout):
        raise SemanticError("layout must be a RunLayout")
    if not isinstance(inputs, ResolvedInput):
        raise SemanticError("inputs must be a ResolvedInput")
    if not isinstance(topology, Topology):
        raise SemanticError("topology must be a Topology")
    if not isinstance(candidate, SolveCandidate):
        raise SemanticError("candidate must be a SolveCandidate")
    if not isinstance(schedule, Schedule):
        raise SemanticError("schedule must be a Schedule")
    if not isinstance(outcome, VerificationOutcome):
        raise SemanticError("outcome must be a VerificationOutcome")
    if overlay is not None and not isinstance(overlay, TuningOverlay):
        raise SemanticError("overlay must be a TuningOverlay or None")
    if not isinstance(accepted, bool):
        raise SemanticError("accepted must be a boolean")
    base = _candidate_base(layout, iteration, selected_best)
    report_path = layout.reports / "{}.validation.json".format(base)
    schedule_path = layout.schedules / "{}.schedule.json".format(base)
    runtime_compatible = outcome.report.runtime_compatible
    emit_xml = outcome.artifact is not None and _offline_valid(outcome)
    if emit_xml:
        suffix = ".xml" if runtime_compatible else ".candidate.xml"
        xml_path: Optional[Path] = layout.schedules / "{}{}".format(base, suffix)
        xml_text = outcome.artifact.xml_text
        xml_digest: Optional[str] = outcome.artifact.sha256
    else:
        xml_path = None
        xml_text = None
        xml_digest = None
    for path in tuple(
        value for value in (xml_path, report_path, schedule_path) if value is not None
    ):
        if path.exists():
            raise FileExistsError("artifact already exists: {}".format(path))

    signature = candidate_signature(schedule, inputs, topology, overlay)
    if outcome.artifact is not None:
        report_object = build_candidate_report(
            candidate,
            inputs,
            topology,
            outcome,
            global_schedule=schedule,
            overlay=overlay,
            applied_strategies=(
                {} if applied_strategies is None else applied_strategies
            ),
            hierarchy_plan=(
                {} if hierarchy_plan is None else hierarchy_plan
            ),
            rejection_reason=rejection_reason,
            selected_best=selected_best,
            tuning_strategy=(
                {} if tuning_strategy is None else tuning_strategy
            ),
        )
        report_payload = json.loads(build_validation_json(report_object))
        binding = (
            report_object.artifact_binding_sha256 if emit_xml else None
        )
    else:
        report_payload = dict(
            _generic_report(candidate, inputs, outcome, signature)
        )
        binding = None
    report_payload.update(
        {
            "artifact_binding_sha256": binding,
            "accepted": accepted,
            "iteration": iteration,
            "selected_best": selected_best,
            "rejection_reason": rejection_reason,
            "xml_path": (
                None
                if xml_path is None
                else str(xml_path.relative_to(layout.root))
            ),
            "schedule_path": str(schedule_path.relative_to(layout.root)),
            "xml_sha256": xml_digest,
        }
    )
    sidecar_text = canonical_json(
        _sidecar_payload(
            inputs,
            candidate,
            schedule,
            signature,
            xml_digest,
            overlay,
        )
    ) + "\n"
    report_text = canonical_json(report_payload) + "\n"
    if xml_path is not None and xml_text is not None:
        atomic_write_text(xml_path, xml_text)
    atomic_write_text(schedule_path, sidecar_text)
    atomic_write_text(report_path, report_text)
    return CandidateArtifact(
        candidate_id=candidate.candidate_id,
        parent_candidate_id=candidate.parent_candidate_id,
        iteration=iteration,
        xml_path=xml_path,
        report_path=report_path,
        schedule_path=schedule_path,
        xml_sha256=xml_digest,
        report_sha256=_sha256(report_text.encode("utf-8")),
        candidate_signature=signature,
        binding_sha256=binding,
        validation=_validation_statuses(outcome),
        runtime_compatible=runtime_compatible,
        accepted=accepted,
        rejection_reason=rejection_reason,
        selected_best=selected_best,
        proven_optimal=candidate.proven_optimal,
        restrictions=candidate.restrictions,
    )


def write_final_alias(
    layout: RunLayout,
    artifact: CandidateArtifact,
) -> Tuple[Path, Path]:
    if not isinstance(layout, RunLayout):
        raise SemanticError("layout must be a RunLayout")
    if not isinstance(artifact, CandidateArtifact):
        raise SemanticError("artifact must be a CandidateArtifact")
    if artifact.xml_path is None:
        raise SemanticError("final alias requires an XML artifact")
    suffix = (
        ".candidate.xml"
        if artifact.xml_path.name.endswith(".candidate.xml")
        else ".xml"
    )
    final_xml = layout.root / "{}_final{}".format(
        layout.artifact_prefix,
        suffix,
    )
    final_report = layout.root / "{}_final.validation.json".format(
        layout.artifact_prefix
    )
    final_schedule = layout.root / "{}_final.schedule.json".format(
        layout.artifact_prefix
    )
    if final_xml.exists() or final_report.exists() or final_schedule.exists():
        raise FileExistsError("final artifact already exists")
    atomic_write_bytes(final_xml, artifact.xml_path.read_bytes())
    atomic_write_bytes(final_report, artifact.report_path.read_bytes())
    atomic_write_bytes(final_schedule, artifact.schedule_path.read_bytes())
    return final_xml, final_report


def write_run_summary(layout: RunLayout, payload: Mapping[str, object]) -> Path:
    if not isinstance(layout, RunLayout):
        raise SemanticError("layout must be a RunLayout")
    if not isinstance(payload, Mapping):
        raise SemanticError("run summary payload must be a mapping")
    atomic_write_text(layout.summary, canonical_json(payload) + "\n")
    return layout.summary
