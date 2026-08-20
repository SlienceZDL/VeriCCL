import math
from dataclasses import dataclass

from vericcl.errors import SemanticError
from vericcl.input.models import ObjectiveMode
from vericcl.topology.model import LinkKey


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


@dataclass(frozen=True)
class RoutingModelStats:
    variable_count: int
    constraint_count: int
    general_constraint_count: int
    build_time_s: float
    optimize_time_s: float

    def __post_init__(self) -> None:
        for field in (
            "variable_count",
            "constraint_count",
            "general_constraint_count",
        ):
            object.__setattr__(
                self,
                field,
                _non_negative_integer(
                    getattr(self, field),
                    "routing_model_stats.{}".format(field),
                ),
            )
        for field in ("build_time_s", "optimize_time_s"):
            object.__setattr__(
                self,
                field,
                _non_negative_number(
                    getattr(self, field),
                    "routing_model_stats.{}".format(field),
                ),
            )


@dataclass(frozen=True)
class RoutePattern:
    template_id: str
    channel_count: int
    objective_mode: ObjectiveMode
    selected_edges: tuple[LinkKey, ...]
    parent_edges: tuple[tuple[int, int], ...]
    model_stats: RoutingModelStats

    def __post_init__(self) -> None:
        _identifier(self.template_id, "route_pattern.template_id")
        object.__setattr__(
            self,
            "channel_count",
            _positive_integer(
                self.channel_count,
                "route_pattern.channel_count",
            ),
        )
        if not isinstance(self.objective_mode, ObjectiveMode):
            raise SemanticError(
                "route_pattern.objective_mode must be an ObjectiveMode"
            )
        if self.objective_mode is ObjectiveMode.AUTO:
            raise SemanticError("route pattern objective must not be AUTO")
        selected_edges = tuple(self.selected_edges)
        if not all(isinstance(edge, LinkKey) for edge in selected_edges):
            raise SemanticError(
                "route_pattern.selected_edges must contain LinkKey values"
            )
        if len(selected_edges) != len(set(selected_edges)):
            raise SemanticError("route pattern selected edges must be unique")
        object.__setattr__(self, "selected_edges", tuple(sorted(selected_edges)))
        try:
            parent_edges = tuple(tuple(edge) for edge in self.parent_edges)
        except TypeError as error:
            raise SemanticError(
                "route_pattern.parent_edges must contain rank pairs"
            ) from error
        if any(len(edge) != 2 for edge in parent_edges):
            raise SemanticError(
                "route_pattern.parent_edges must contain rank pairs"
            )
        for src_rank, dst_rank in parent_edges:
            LinkKey(src_rank, dst_rank)
        if len(parent_edges) != len(set(parent_edges)):
            raise SemanticError("route pattern parent edges must be unique")
        object.__setattr__(self, "parent_edges", tuple(sorted(parent_edges)))
        if not isinstance(self.model_stats, RoutingModelStats):
            raise SemanticError(
                "route_pattern.model_stats must be RoutingModelStats"
            )
