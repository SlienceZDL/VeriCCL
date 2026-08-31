import math
from dataclasses import dataclass
from typing import Tuple

from vericcl.errors import SemanticError
from vericcl.input.models import ObjectiveMode
from vericcl.solver.model import SolveStatus, SolverMetrics


Edge = Tuple[int, int]
MemberPath = Tuple[str, Tuple[Edge, ...]]


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SemanticError("{} must be a non-empty string".format(field))
    return value


def _non_negative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SemanticError("{} must be a non-negative integer".format(field))
    return value


def _positive_integer(value: object, field: str) -> int:
    result = _non_negative_integer(value, field)
    if result < 1:
        raise SemanticError("{} must be a positive integer".format(field))
    return result


def _non_negative_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticError("{} must be a number".format(field))
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise SemanticError("{} must be finite and non-negative".format(field))
    return result


def _edge(value: object, field: str) -> Edge:
    try:
        values = tuple(value)
    except TypeError as error:
        raise SemanticError("{} must be a rank pair".format(field)) from error
    if len(values) != 2:
        raise SemanticError("{} must be a rank pair".format(field))
    src, dst = values
    _non_negative_integer(src, field)
    _non_negative_integer(dst, field)
    if src == dst:
        raise SemanticError("{} must connect distinct ranks".format(field))
    return src, dst


@dataclass(frozen=True)
class RoutingModelStats:
    variable_count: int
    constraint_count: int
    general_constraint_count: int
    build_time_s: float
    optimize_time_s: float

    def __post_init__(self) -> None:
        _non_negative_integer(
            self.variable_count,
            "routing_model_stats.variable_count",
        )
        _non_negative_integer(
            self.constraint_count,
            "routing_model_stats.constraint_count",
        )
        _non_negative_integer(
            self.general_constraint_count,
            "routing_model_stats.general_constraint_count",
        )
        object.__setattr__(
            self,
            "build_time_s",
            _non_negative_number(
                self.build_time_s,
                "routing_model_stats.build_time_s",
            ),
        )
        object.__setattr__(
            self,
            "optimize_time_s",
            _non_negative_number(
                self.optimize_time_s,
                "routing_model_stats.optimize_time_s",
            ),
        )


@dataclass(frozen=True)
class RoutePattern:
    template_id: str
    channel_count: int
    objective_mode: ObjectiveMode
    selected_edges: Tuple[Edge, ...]
    member_paths: Tuple[MemberPath, ...]
    metrics: SolverMetrics
    model_stats: RoutingModelStats

    def __post_init__(self) -> None:
        _identifier(self.template_id, "route_pattern.template_id")
        _positive_integer(self.channel_count, "route_pattern.channel_count")
        if not isinstance(self.objective_mode, ObjectiveMode):
            raise SemanticError(
                "route_pattern.objective_mode must be an ObjectiveMode"
            )
        if self.objective_mode is ObjectiveMode.AUTO:
            raise SemanticError(
                "AUTO must be resolved before creating a RoutePattern"
            )
        try:
            selected = tuple(
                _edge(value, "route_pattern.selected_edges")
                for value in self.selected_edges
            )
        except TypeError as error:
            raise SemanticError(
                "route_pattern.selected_edges must be iterable"
            ) from error
        if not selected:
            raise SemanticError("route_pattern.selected_edges must not be empty")
        if len(selected) != len(set(selected)):
            raise SemanticError("route_pattern.selected_edges must be unique")
        selected = tuple(sorted(selected))
        object.__setattr__(self, "selected_edges", selected)
        try:
            raw_paths = tuple(self.member_paths)
        except TypeError as error:
            raise SemanticError(
                "route_pattern.member_paths must be iterable"
            ) from error
        normalized = []
        for value in raw_paths:
            try:
                member_id, raw_path = tuple(value)
            except (TypeError, ValueError) as error:
                raise SemanticError(
                    "route_pattern.member_paths must contain ID/path pairs"
                ) from error
            _identifier(member_id, "route_pattern.member_paths member ID")
            try:
                path = tuple(
                    _edge(edge, "route_pattern.member_paths path edge")
                    for edge in raw_path
                )
            except TypeError as error:
                raise SemanticError(
                    "route_pattern.member_paths paths must be iterable"
                ) from error
            if not path:
                raise SemanticError(
                    "route_pattern.member_paths paths must not be empty"
                )
            for first, second in zip(path, path[1:]):
                if first[1] != second[0]:
                    raise SemanticError(
                        "route_pattern member paths must be continuous"
                    )
            nodes = (path[0][0],) + tuple(edge[1] for edge in path)
            if len(nodes) != len(set(nodes)):
                raise SemanticError("route_pattern member path contains a cycle")
            if any(edge not in selected for edge in path):
                raise SemanticError(
                    "route_pattern member path is outside the selected tree"
                )
            normalized.append((member_id, path))
        member_ids = tuple(member_id for member_id, _ in normalized)
        if not member_ids:
            raise SemanticError("route_pattern.member_paths must not be empty")
        if len(member_ids) != len(set(member_ids)):
            raise SemanticError("route_pattern member IDs must be unique")
        used_edges = {
            edge for _, path in normalized for edge in path
        }
        if used_edges != set(selected):
            raise SemanticError(
                "route_pattern selected tree must contain exactly the path edges"
            )
        object.__setattr__(
            self,
            "member_paths",
            tuple(sorted(normalized, key=lambda item: item[0])),
        )
        if not isinstance(self.metrics, SolverMetrics):
            raise SemanticError("route_pattern.metrics must be SolverMetrics")
        if self.metrics.status not in {
            SolveStatus.OPTIMAL,
            SolveStatus.FEASIBLE,
            SolveStatus.TIME_LIMIT,
        }:
            raise SemanticError("route_pattern.metrics must describe an incumbent")
        if not isinstance(self.model_stats, RoutingModelStats):
            raise SemanticError(
                "route_pattern.model_stats must be RoutingModelStats"
            )
