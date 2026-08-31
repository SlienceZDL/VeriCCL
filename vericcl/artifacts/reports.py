from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
import math
from typing import Mapping, Optional

from vericcl.artifacts.hashing import (
    artifact_binding_sha256,
    candidate_signature,
)
from vericcl.errors import SemanticError
from vericcl.input.json_codec import canonical_json
from vericcl.input.models import ResolvedInput
from vericcl.semantics.atom import Schedule
from vericcl.solver.model import SearchDiagnostics, SolveCandidate
from vericcl.topology.model import Topology
from vericcl.tuning.model import TuningOverlay
from vericcl.verification.model import ValidationReport
from vericcl.verification.pipeline import VerificationOutcome
from vericcl.xml.recommendations import recommend_runtime_compatible_inputs


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SemanticError("{} must be a mapping".format(field))
    if not all(isinstance(key, str) and key for key in value):
        raise SemanticError("{} keys must be strings".format(field))
    return MappingProxyType(dict(value))


def _non_negative_time(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticError("{} must be numeric".format(field))
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise SemanticError("{} must be finite and non-negative".format(field))
    return result


def _overlay(value: Optional[TuningOverlay]) -> Mapping[str, object]:
    if value is None:
        return MappingProxyType({})
    return MappingProxyType(
        {
            "overlay_id": value.overlay_id,
            "parent_candidate_id": value.parent_candidate_id,
            "channel_count": value.channel_count,
            "path_weights": value.path_weights,
            "temporary_forbidden": tuple(
                (
                    item.slice_id,
                    item.src_rank,
                    item.dst_rank,
                    item.stage_id,
                )
                for item in sorted(
                    value.temporary_forbidden,
                    key=lambda item: (
                        item.slice_id,
                        item.src_rank,
                        item.dst_rank,
                        item.stage_id,
                    ),
                )
            ),
            "batch_size": value.batch_size,
            "tree_roots": value.tree_roots,
            "tree_edges": value.tree_edges,
            "lane_order": value.lane_order,
            "milp_parameters": value.milp_parameters,
            "warm_start_candidate_id": value.warm_start_candidate_id,
            "resolve_scope": value.resolve_scope,
            "hierarchy_template": value.hierarchy_template,
        }
    )


def _strategies(inputs: ResolvedInput) -> Mapping[str, bool]:
    value = inputs.strategies
    return MappingProxyType(
        {
            "hierarchy": value.hierarchy,
            "symmetry": value.symmetry,
            "shortest_paths": value.shortest_paths,
            "batching": value.batching,
            "constructive_trees": value.constructive_trees,
            "milp": value.milp,
        }
    )


def _strategy_parameters(inputs: ResolvedInput) -> Mapping[str, object]:
    hyper = inputs.hyperparameters
    solver = inputs.solver
    return MappingProxyType(
        {
            "total_size_bytes": hyper.total_size_bytes,
            "slice_size_bytes": hyper.slice_size_bytes,
            "objective_mode": hyper.objective_mode.value,
            "max_calibration_channels": hyper.max_calibration_channels,
            "min_expected_improvement": hyper.min_expected_improvement,
            "min_tuning_improvement": hyper.min_tuning_improvement,
            "max_tuning_iterations": hyper.max_tuning_iterations,
            "total_verification_timeout_s": (
                hyper.total_verification_timeout_s
            ),
            "total_solve_timeout_s": solver.total_solve_timeout_s,
            "per_model_timeout_s": solver.per_model_timeout_s,
            "mip_gap": solver.mip_gap,
            "require_proven_optimal": solver.require_proven_optimal,
            "solver_seed": solver.solver_seed,
            "max_channels": solver.max_channels,
            "max_threads_per_model": solver.max_threads_per_model,
            "max_parallel_models": solver.max_parallel_models,
        }
    )


def _reproducibility(candidate: SolveCandidate) -> Mapping[str, object]:
    metrics = candidate.metrics
    return MappingProxyType(
        {
            "solver_seed": metrics.solver_seed,
            "thread_count": metrics.thread_count,
            "solver_name": metrics.solver_name,
            "solver_version": metrics.solver_version,
            "deterministic_artifacts": True,
            "limits": (
                "environment_signature",
                "hardware_measurement",
                "parallel_solver_execution",
                "solver_version",
            ),
        }
    )


def _buffer_plan(outcome: VerificationOutcome) -> Mapping[str, object]:
    if outcome.artifact is None:
        return MappingProxyType({})
    plan = outcome.artifact.buffer_plan
    rank_chunks = lambda values: {
        "r{:08d}".format(rank): count
        for rank, count in values.items()
    }
    return MappingProxyType(
        {
            "slice_count": plan.slice_count,
            "local_copy_count": len(plan.local_copies),
            "input_chunks": rank_chunks(plan.i_chunks),
            "output_chunks": rank_chunks(plan.o_chunks),
            "scratch_chunks": rank_chunks(plan.s_chunks),
            "value_count": len(plan.value_locations),
        }
    )


@dataclass(frozen=True)
class CandidateReport:
    schema_version: str
    candidate_id: str
    normalized_input_sha256: str
    topology_signature: str
    candidate_signature: str
    artifact_binding_sha256: str
    requested_strategies: Mapping[str, object]
    applied_strategies: Mapping[str, object]
    strategy_parameters: Mapping[str, object]
    overlay: Mapping[str, object]
    hierarchy_plan: Mapping[str, object]
    channel_count: int
    buffer_plan: Mapping[str, object]
    solver_metrics: object
    validation: ValidationReport
    lineage: Mapping[str, object]
    rejection_reason: Optional[str]
    selected_best: bool
    proven_optimal: bool
    search_space_restricted: bool
    runtime_compatible: bool
    xml_sha256: str
    bdd_evidence: Mapping[str, object]
    simulation_evidence: Mapping[str, object]
    tuning_strategy: Mapping[str, object]
    runtime_recommendations: tuple
    reproducibility: Mapping[str, object]
    planning_mode: str = "unknown"
    requested_problem_count: int = 0
    routing_unit_count: int = 0
    template_count: int = 0
    template_member_count: int = 0
    route_model_count: int = 0
    fallback_member_model_count: int = 0
    search_model_count_total: int = 0
    route_model_build_time_s: float = 0.0
    route_model_optimize_time_s: float = 0.0
    template_expansion_time_s: float = 0.0
    global_scheduling_time_s: float = 0.0
    model_variables_max: int = 0
    model_constraints_max: int = 0
    model_general_constraints_max: int = 0
    verification_time_s: float = 0.0
    cache_hit: bool = False

    def __post_init__(self) -> None:
        for field in (
            "requested_strategies",
            "applied_strategies",
            "strategy_parameters",
            "overlay",
            "hierarchy_plan",
            "lineage",
            "bdd_evidence",
            "simulation_evidence",
            "tuning_strategy",
            "reproducibility",
        ):
            object.__setattr__(
                self,
                field,
                _mapping(getattr(self, field), field),
            )
        if not isinstance(self.validation, ValidationReport):
            raise SemanticError("candidate report validation is invalid")
        if not isinstance(self.planning_mode, str) or not self.planning_mode:
            raise SemanticError("candidate report planning_mode is invalid")
        SearchDiagnostics.from_mapping(
            {
                name: getattr(self, name)
                for name in (
                    "requested_problem_count",
                    "routing_unit_count",
                    "template_count",
                    "template_member_count",
                    "route_model_count",
                    "fallback_member_model_count",
                    "search_model_count_total",
                    "route_model_build_time_s",
                    "route_model_optimize_time_s",
                    "template_expansion_time_s",
                    "global_scheduling_time_s",
                    "model_variables_max",
                    "model_constraints_max",
                    "model_general_constraints_max",
                )
            }
        )
        object.__setattr__(
            self,
            "verification_time_s",
            _non_negative_time(
                self.verification_time_s,
                "candidate report verification_time_s",
            ),
        )
        if not isinstance(self.cache_hit, bool):
            raise SemanticError("candidate report cache_hit must be a boolean")


def build_candidate_report(
    candidate: SolveCandidate,
    inputs: ResolvedInput,
    topology: Topology,
    outcome: VerificationOutcome,
    *,
    global_schedule: Optional[Schedule] = None,
    overlay: Optional[TuningOverlay],
    applied_strategies: Mapping[str, object],
    hierarchy_plan: Mapping[str, object],
    rejection_reason: Optional[str],
    selected_best: bool,
    tuning_strategy: Mapping[str, object],
    diagnostics: Optional[SearchDiagnostics] = None,
    verification_time_s: float = 0.0,
    cache_hit: bool = False,
) -> CandidateReport:
    if not isinstance(candidate, SolveCandidate):
        raise SemanticError("candidate must be a SolveCandidate")
    if not isinstance(inputs, ResolvedInput):
        raise SemanticError("inputs must be a ResolvedInput")
    if not isinstance(topology, Topology):
        raise SemanticError("topology must be a Topology")
    if not isinstance(outcome, VerificationOutcome):
        raise SemanticError("outcome must be a VerificationOutcome")
    hierarchy_value = _mapping(hierarchy_plan, "hierarchy_plan")
    if outcome.artifact is None:
        raise SemanticError("candidate report requires an XML artifact")
    diagnostics_value = (
        SearchDiagnostics() if diagnostics is None else diagnostics
    )
    if not isinstance(diagnostics_value, SearchDiagnostics):
        raise SemanticError("candidate report diagnostics are invalid")
    if global_schedule is not None and not isinstance(
        global_schedule,
        Schedule,
    ):
        raise SemanticError("global_schedule must be a Schedule or None")
    schedules = tuple(candidate.node_schedules.values())
    if global_schedule is None and len(schedules) != 1:
        raise SemanticError(
            "candidate report requires one bound global schedule"
        )
    bound_schedule = (
        global_schedule if global_schedule is not None else schedules[0]
    )
    signature = candidate_signature(
        bound_schedule,
        inputs,
        topology,
        overlay,
    )
    xml_sha256 = outcome.artifact.sha256
    requested = _strategies(inputs)
    unknown_applied = set(applied_strategies) - set(requested)
    if unknown_applied:
        raise SemanticError("applied strategies contain an unknown field")
    applied = {
        name: applied_strategies.get(name, requested_value)
        for name, requested_value in requested.items()
    }
    return CandidateReport(
        schema_version="1",
        candidate_id=candidate.candidate_id,
        normalized_input_sha256=inputs.input_sha256,
        topology_signature=topology.isomorphism_signature,
        candidate_signature=signature,
        artifact_binding_sha256=artifact_binding_sha256(
            inputs.input_sha256,
            signature,
            xml_sha256,
        ),
        requested_strategies=requested,
        applied_strategies=applied,
        strategy_parameters=_strategy_parameters(inputs),
        overlay=_overlay(overlay),
        hierarchy_plan=hierarchy_value,
        channel_count=(
            overlay.channel_count
            if overlay is not None and overlay.channel_count is not None
            else candidate.channel_count
        ),
        buffer_plan=_buffer_plan(outcome),
        solver_metrics=candidate.metrics,
        validation=outcome.report,
        lineage={
            "candidate_id": candidate.candidate_id,
            "parent_candidate_id": candidate.parent_candidate_id,
        },
        rejection_reason=rejection_reason,
        selected_best=selected_best,
        proven_optimal=candidate.proven_optimal,
        search_space_restricted=candidate.search_space_restricted,
        runtime_compatible=outcome.report.runtime_compatible,
        xml_sha256=xml_sha256,
        bdd_evidence=dict(outcome.report.bdd.evidence),
        simulation_evidence=dict(outcome.report.simulation.evidence),
        tuning_strategy=tuning_strategy,
        runtime_recommendations=recommend_runtime_compatible_inputs(
            inputs,
            outcome.artifact,
        ),
        reproducibility=_reproducibility(candidate),
        planning_mode=hierarchy_value.get("planning_mode", "unknown"),
        requested_problem_count=diagnostics_value.requested_problem_count,
        routing_unit_count=diagnostics_value.routing_unit_count,
        template_count=diagnostics_value.template_count,
        template_member_count=diagnostics_value.template_member_count,
        route_model_count=diagnostics_value.route_model_count,
        fallback_member_model_count=(
            diagnostics_value.fallback_member_model_count
        ),
        search_model_count_total=diagnostics_value.search_model_count_total,
        route_model_build_time_s=(
            diagnostics_value.route_model_build_time_s
        ),
        route_model_optimize_time_s=(
            diagnostics_value.route_model_optimize_time_s
        ),
        template_expansion_time_s=(
            diagnostics_value.template_expansion_time_s
        ),
        global_scheduling_time_s=(
            diagnostics_value.global_scheduling_time_s
        ),
        model_variables_max=diagnostics_value.model_variables_max,
        model_constraints_max=diagnostics_value.model_constraints_max,
        model_general_constraints_max=(
            diagnostics_value.model_general_constraints_max
        ),
        verification_time_s=verification_time_s,
        cache_hit=cache_hit,
    )


def build_validation_json(report: object) -> str:
    if not isinstance(report, (CandidateReport, ValidationReport)):
        raise SemanticError(
            "report must be a CandidateReport or ValidationReport"
        )
    return canonical_json(report)
