import math
from dataclasses import dataclass
from typing import Optional, Tuple

from vericcl.errors import SemanticError
from vericcl.input.models import ForbiddenTransfer


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SemanticError("{} must be a non-empty string".format(field))
    return value


def _optional_identifier(value: object, field: str) -> Optional[str]:
    if value is None:
        return None
    return _identifier(value, field)


def _integer(value: object, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SemanticError("{} must be an integer".format(field))
    if value < minimum:
        raise SemanticError("{} must be at least {}".format(field, minimum))
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticError("{} must be a number".format(field))
    result = float(value)
    if not math.isfinite(result):
        raise SemanticError("{} must be finite".format(field))
    return result


def _unique_identifiers(value: object, field: str) -> Tuple[str, ...]:
    try:
        items = tuple(value)
    except TypeError as error:
        raise SemanticError("{} must be iterable".format(field)) from error
    for item in items:
        _identifier(item, field)
    if len(items) != len(set(items)):
        raise SemanticError("{} must be unique".format(field))
    return tuple(sorted(items))


def _weighted_pairs(value: object, field: str) -> Tuple[Tuple[str, float], ...]:
    try:
        pairs = tuple(value)
    except TypeError as error:
        raise SemanticError("{} must be iterable".format(field)) from error
    normalized = []
    for pair in pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise SemanticError("{} entries must be pairs".format(field))
        key, weight = pair
        normalized.append((_identifier(key, field), _number(weight, field)))
    keys = [key for key, _ in normalized]
    if len(keys) != len(set(keys)):
        raise SemanticError("{} keys must be unique".format(field))
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class TuningOverlay:
    overlay_id: str
    parent_candidate_id: Optional[str]
    channel_count: Optional[int] = None
    path_weights: Tuple[Tuple[str, float], ...] = ()
    temporary_forbidden: frozenset = frozenset()
    batch_size: Optional[int] = None
    tree_roots: Tuple[Tuple[int, int], ...] = ()
    tree_edges: Tuple[Tuple[int, int, int], ...] = ()
    lane_order: Tuple[Tuple[str, str], ...] = ()
    milp_parameters: Tuple[Tuple[str, float], ...] = ()
    warm_start_candidate_id: Optional[str] = None
    resolve_scope: Tuple[str, ...] = ()
    hierarchy_template: Optional[str] = None

    def __post_init__(self) -> None:
        _identifier(self.overlay_id, "tuning_overlay.overlay_id")
        _optional_identifier(
            self.parent_candidate_id,
            "tuning_overlay.parent_candidate_id",
        )
        if self.channel_count is not None:
            _integer(
                self.channel_count,
                "tuning_overlay.channel_count",
                minimum=1,
            )
        if self.batch_size is not None:
            _integer(self.batch_size, "tuning_overlay.batch_size", minimum=1)
        forbidden = frozenset(self.temporary_forbidden)
        if not all(isinstance(item, ForbiddenTransfer) for item in forbidden):
            raise SemanticError(
                "tuning_overlay.temporary_forbidden must contain ForbiddenTransfer values"
            )
        object.__setattr__(self, "temporary_forbidden", forbidden)
        object.__setattr__(
            self,
            "path_weights",
            _weighted_pairs(self.path_weights, "tuning_overlay.path_weights"),
        )
        object.__setattr__(
            self,
            "milp_parameters",
            _weighted_pairs(
                self.milp_parameters,
                "tuning_overlay.milp_parameters",
            ),
        )
        object.__setattr__(
            self,
            "tree_roots",
            self._integer_tuples(self.tree_roots, "tree_roots", 2),
        )
        object.__setattr__(
            self,
            "tree_edges",
            self._integer_tuples(self.tree_edges, "tree_edges", 3),
        )
        object.__setattr__(
            self,
            "lane_order",
            self._lane_order(self.lane_order),
        )
        _optional_identifier(
            self.warm_start_candidate_id,
            "tuning_overlay.warm_start_candidate_id",
        )
        object.__setattr__(
            self,
            "resolve_scope",
            _unique_identifiers(
                self.resolve_scope,
                "tuning_overlay.resolve_scope",
            ),
        )
        _optional_identifier(
            self.hierarchy_template,
            "tuning_overlay.hierarchy_template",
        )

    @staticmethod
    def _integer_tuples(
        value: object,
        field: str,
        arity: int,
    ) -> tuple:
        try:
            items = tuple(value)
        except TypeError as error:
            raise SemanticError(
                "tuning_overlay.{} must be iterable".format(field)
            ) from error
        normalized = []
        for item in items:
            if not isinstance(item, tuple) or len(item) != arity:
                raise SemanticError(
                    "tuning_overlay.{} entries must have {} integers".format(
                        field,
                        arity,
                    )
                )
            normalized.append(
                tuple(
                    _integer(part, "tuning_overlay.{}".format(field))
                    for part in item
                )
            )
        if len(normalized) != len(set(normalized)):
            raise SemanticError(
                "tuning_overlay.{} must be unique".format(field)
            )
        return tuple(sorted(normalized))

    @staticmethod
    def _lane_order(value: object) -> Tuple[Tuple[str, str], ...]:
        try:
            items = tuple(value)
        except TypeError as error:
            raise SemanticError("tuning_overlay.lane_order must be iterable") from error
        normalized = []
        for item in items:
            if not isinstance(item, tuple) or len(item) != 2:
                raise SemanticError(
                    "tuning_overlay.lane_order entries must be pairs"
                )
            predecessor, successor = item
            _identifier(predecessor, "tuning_overlay.lane_order")
            _identifier(successor, "tuning_overlay.lane_order")
            if predecessor == successor:
                raise SemanticError(
                    "tuning_overlay.lane_order cannot order an item against itself"
                )
            normalized.append((predecessor, successor))
        if len(normalized) != len(set(normalized)):
            raise SemanticError("tuning_overlay.lane_order must be unique")
        return tuple(sorted(normalized))
