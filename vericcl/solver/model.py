import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from vericcl.errors import SemanticError
from vericcl.input.models import ObjectiveMode, ResolvedInput
from vericcl.planner.model import PlanDAG
from vericcl.semantics.atom import Schedule
from vericcl.topology.model import Topology
from vericcl.tuning.model import TuningOverlay


class SolveStatus(str, Enum):
    NOT_RUN = "not_run"
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    TIME_LIMIT = "time_limit"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SemanticError("{} must be a non-empty string".format(field))
    return value


def _integer(value: object, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SemanticError("{} must be an integer".format(field))
    if value < minimum:
        raise SemanticError("{} must be at least {}".format(field, minimum))
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


@dataclass(frozen=True)
class SolverMetrics:
    status: SolveStatus
    objective_values: Tuple[float, ...]
    best_bound: float
    mip_gap: float
    within_requested_gap: bool
    solve_time_s: float
    model_count: int
    operation_count: int
    hop_count: int
    makespan_us: float
    maximum_normalized_resource_load: float
    solver_name: str
    solver_version: str
    solver_seed: int
    thread_count: int
    termination_reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, SolveStatus):
            raise SemanticError("solver_metrics.status must be a SolveStatus")
        values = tuple(
            _number(value, "solver_metrics.objective_values")
            for value in self.objective_values
        )
        object.__setattr__(self, "objective_values", values)
        object.__setattr__(
            self,
            "best_bound",
            _number(self.best_bound, "solver_metrics.best_bound"),
        )
        object.__setattr__(
            self,
            "mip_gap",
            _number(self.mip_gap, "solver_metrics.mip_gap"),
        )
        if not isinstance(self.within_requested_gap, bool):
            raise SemanticError(
                "solver_metrics.within_requested_gap must be a boolean"
            )
        object.__setattr__(
            self,
            "solve_time_s",
            _number(self.solve_time_s, "solver_metrics.solve_time_s"),
        )
        _integer(self.model_count, "solver_metrics.model_count")
        _integer(self.operation_count, "solver_metrics.operation_count")
        _integer(self.hop_count, "solver_metrics.hop_count")
        object.__setattr__(
            self,
            "makespan_us",
            _number(self.makespan_us, "solver_metrics.makespan_us"),
        )
        object.__setattr__(
            self,
            "maximum_normalized_resource_load",
            _number(
                self.maximum_normalized_resource_load,
                "solver_metrics.maximum_normalized_resource_load",
            ),
        )
        _identifier(self.solver_name, "solver_metrics.solver_name")
        _identifier(self.solver_version, "solver_metrics.solver_version")
        _integer(self.solver_seed, "solver_metrics.solver_seed")
        _integer(self.thread_count, "solver_metrics.thread_count")
        _identifier(
            self.termination_reason,
            "solver_metrics.termination_reason",
        )


@dataclass(frozen=True)
class SolveRequest:
    inputs: ResolvedInput
    topology: Topology
    plan: PlanDAG
    overlay: Optional[TuningOverlay] = None
    solver_version: str = "unknown"
    model_version: str = "1"
    environment_signature: str = "unknown"

    def __post_init__(self) -> None:
        if not isinstance(self.inputs, ResolvedInput):
            raise SemanticError("solve_request.inputs must be a ResolvedInput")
        if not isinstance(self.topology, Topology):
            raise SemanticError("solve_request.topology must be a Topology")
        if not isinstance(self.plan, PlanDAG):
            raise SemanticError("solve_request.plan must be a PlanDAG")
        if self.overlay is not None and not isinstance(
            self.overlay,
            TuningOverlay,
        ):
            raise SemanticError(
                "solve_request.overlay must be a TuningOverlay or None"
            )
        if (
            self.inputs.rank_count != self.topology.rank_count
            or self.inputs.rank_count != self.plan.rank_count
        ):
            raise SemanticError("solve request rank counts must agree")
        if self.plan.slice_count != self.inputs.hyperparameters.slice_count:
            raise SemanticError("solve request slice counts must agree")
        if self.plan.collective != self.inputs.collective:
            raise SemanticError("solve request collective specifications must agree")
        _identifier(self.solver_version, "solve_request.solver_version")
        _identifier(self.model_version, "solve_request.model_version")
        _identifier(
            self.environment_signature,
            "solve_request.environment_signature",
        )


@dataclass(frozen=True)
class SolveCandidate:
    candidate_id: str
    node_schedules: Mapping[str, Schedule]
    objective_mode: ObjectiveMode
    channel_count: int
    metrics: SolverMetrics
    selected_best: bool
    proven_optimal: bool
    search_space_restricted: bool
    restrictions: Tuple[str, ...]
    parent_candidate_id: Optional[str]

    def __post_init__(self) -> None:
        _identifier(self.candidate_id, "solve_candidate.candidate_id")
        try:
            schedules = dict(self.node_schedules)
        except (TypeError, ValueError) as error:
            raise SemanticError(
                "solve_candidate.node_schedules must be a mapping"
            ) from error
        for node_id, schedule in schedules.items():
            _identifier(node_id, "solve_candidate.node_schedules")
            if not isinstance(schedule, Schedule):
                raise SemanticError(
                    "solve_candidate.node_schedules must contain Schedule values"
                )
        object.__setattr__(
            self,
            "node_schedules",
            MappingProxyType(dict(sorted(schedules.items()))),
        )
        if not isinstance(self.objective_mode, ObjectiveMode):
            raise SemanticError(
                "solve_candidate.objective_mode must be an ObjectiveMode"
            )
        _integer(self.channel_count, "solve_candidate.channel_count", minimum=1)
        if not isinstance(self.metrics, SolverMetrics):
            raise SemanticError(
                "solve_candidate.metrics must be SolverMetrics"
            )
        for field, value in (
            ("selected_best", self.selected_best),
            ("proven_optimal", self.proven_optimal),
            ("search_space_restricted", self.search_space_restricted),
        ):
            if not isinstance(value, bool):
                raise SemanticError(
                    "solve_candidate.{} must be a boolean".format(field)
                )
        restrictions = tuple(self.restrictions)
        for restriction in restrictions:
            _identifier(restriction, "solve_candidate.restrictions")
        if len(restrictions) != len(set(restrictions)):
            raise SemanticError("solve_candidate.restrictions must be unique")
        restrictions = tuple(sorted(restrictions))
        object.__setattr__(self, "restrictions", restrictions)
        if self.search_space_restricted != bool(restrictions):
            raise SemanticError(
                "solve_candidate restrictions and search_space_restricted must agree"
            )
        if self.proven_optimal and self.metrics.status is not SolveStatus.OPTIMAL:
            raise SemanticError(
                "proven_optimal requires solver status OPTIMAL"
            )
        if self.proven_optimal and self.search_space_restricted:
            raise SemanticError(
                "a restricted search space cannot prove global optimality"
            )
        if self.parent_candidate_id is not None:
            _identifier(
                self.parent_candidate_id,
                "solve_candidate.parent_candidate_id",
            )


@dataclass(frozen=True)
class SolveResult:
    status: SolveStatus
    candidates: Tuple[SolveCandidate, ...]
    selected_candidate_id: Optional[str]
    cache_hit: bool
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, SolveStatus):
            raise SemanticError("solve_result.status must be a SolveStatus")
        candidates = tuple(self.candidates)
        if not all(isinstance(item, SolveCandidate) for item in candidates):
            raise SemanticError(
                "solve_result.candidates must contain SolveCandidate values"
            )
        candidate_ids = tuple(item.candidate_id for item in candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise SemanticError("solve result candidate IDs must be unique")
        object.__setattr__(self, "candidates", candidates)
        if self.selected_candidate_id is not None:
            _identifier(
                self.selected_candidate_id,
                "solve_result.selected_candidate_id",
            )
            if self.selected_candidate_id not in candidate_ids:
                raise SemanticError(
                    "solve result selected candidate does not exist"
                )
        if not isinstance(self.cache_hit, bool):
            raise SemanticError("solve_result.cache_hit must be a boolean")
        if not isinstance(self.message, str):
            raise SemanticError("solve_result.message must be a string")

    @property
    def selected_candidate(self) -> Optional[SolveCandidate]:
        for candidate in self.candidates:
            if candidate.candidate_id == self.selected_candidate_id:
                return candidate
        return None
