from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import FrozenSet, Mapping, Optional

from vericcl.errors import SemanticError


class CollectiveKind(str, Enum):
    BROADCAST = "broadcast"
    REDUCE = "reduce"
    SCATTER = "scatter"
    GATHER = "gather"
    ALL_GATHER = "allgather"
    ALL_REDUCE = "allreduce"
    ALL_TO_ALL = "alltoall"
    REDUCE_SCATTER = "reduce_scatter"


@dataclass(frozen=True)
class CollectiveSpec:
    kind: CollectiveKind
    datatype: str
    reduction_op: Optional[str] = None
    root: Optional[int] = None
    inplace: bool = False


@dataclass(frozen=True, order=True)
class OutputSlot:
    rank: int
    offset: int

    def __post_init__(self) -> None:
        _positive_or_zero(self.rank, "output_slot.rank")
        _positive_or_zero(self.offset, "output_slot.offset")


def _positive_or_zero(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SemanticError("{} must be an integer".format(field))
    if value < 0:
        raise SemanticError("{} must be non-negative".format(field))
    return value


def _positive(value: object, field: str) -> int:
    normalized = _positive_or_zero(value, field)
    if normalized == 0:
        raise SemanticError("{} must be positive".format(field))
    return normalized


def _validate_output_problem(
    spec: CollectiveSpec,
    rank_count: int,
    slice_count: int,
) -> None:
    if not isinstance(spec, CollectiveSpec):
        raise SemanticError("spec must be a CollectiveSpec")
    _positive(rank_count, "rank_count")
    _positive(slice_count, "slice_count")
    rooted = {
        CollectiveKind.BROADCAST,
        CollectiveKind.REDUCE,
        CollectiveKind.SCATTER,
        CollectiveKind.GATHER,
    }
    if spec.kind in rooted:
        if isinstance(spec.root, bool) or not isinstance(spec.root, int):
            raise SemanticError("{} requires an integer root".format(spec.kind.value))
        if spec.root < 0 or spec.root >= rank_count:
            raise SemanticError("collective root is outside the rank range")
    elif spec.root is not None:
        raise SemanticError("{} must not define root".format(spec.kind.value))
    if spec.kind in {
        CollectiveKind.SCATTER,
        CollectiveKind.ALL_TO_ALL,
        CollectiveKind.REDUCE_SCATTER,
    }:
        if slice_count % rank_count != 0:
            raise SemanticError("slice count must be divisible by rank count")


def _aggregate_contributors(
    logical_address: int,
    rank_count: int,
    slice_count: int,
) -> FrozenSet[int]:
    return frozenset(
        rank * slice_count + logical_address for rank in range(rank_count)
    )


def required_outputs(
    spec: CollectiveSpec,
    rank_count: int,
    slice_count: int,
) -> Mapping[OutputSlot, FrozenSet[int]]:
    _validate_output_problem(spec, rank_count, slice_count)
    outputs = {}
    if spec.kind is CollectiveKind.BROADCAST:
        for rank in range(rank_count):
            for logical_address in range(slice_count):
                contributor = spec.root * slice_count + logical_address
                outputs[OutputSlot(rank, logical_address)] = frozenset(
                    {contributor}
                )
    elif spec.kind is CollectiveKind.REDUCE:
        for logical_address in range(slice_count):
            outputs[OutputSlot(spec.root, logical_address)] = (
                _aggregate_contributors(
                    logical_address,
                    rank_count,
                    slice_count,
                )
            )
    elif spec.kind is CollectiveKind.SCATTER:
        quotient = slice_count // rank_count
        for logical_address in range(slice_count):
            owner = logical_address // quotient
            offset = logical_address % quotient
            contributor = spec.root * slice_count + logical_address
            outputs[OutputSlot(owner, offset)] = frozenset({contributor})
    elif spec.kind is CollectiveKind.GATHER:
        for source_rank in range(rank_count):
            for logical_address in range(slice_count):
                contributor = source_rank * slice_count + logical_address
                outputs[OutputSlot(spec.root, contributor)] = frozenset(
                    {contributor}
                )
    elif spec.kind is CollectiveKind.ALL_GATHER:
        for rank in range(rank_count):
            for source_rank in range(rank_count):
                for logical_address in range(slice_count):
                    contributor = source_rank * slice_count + logical_address
                    outputs[OutputSlot(rank, contributor)] = frozenset(
                        {contributor}
                    )
    elif spec.kind is CollectiveKind.ALL_REDUCE:
        for rank in range(rank_count):
            for logical_address in range(slice_count):
                outputs[OutputSlot(rank, logical_address)] = (
                    _aggregate_contributors(
                        logical_address,
                        rank_count,
                        slice_count,
                    )
                )
    elif spec.kind is CollectiveKind.ALL_TO_ALL:
        quotient = slice_count // rank_count
        for source_rank in range(rank_count):
            for logical_address in range(slice_count):
                destination = logical_address // quotient
                offset = source_rank * quotient + logical_address % quotient
                contributor = source_rank * slice_count + logical_address
                outputs[OutputSlot(destination, offset)] = frozenset(
                    {contributor}
                )
    elif spec.kind is CollectiveKind.REDUCE_SCATTER:
        quotient = slice_count // rank_count
        for logical_address in range(slice_count):
            owner = logical_address // quotient
            offset = logical_address % quotient
            outputs[OutputSlot(owner, offset)] = _aggregate_contributors(
                logical_address,
                rank_count,
                slice_count,
            )
    else:
        raise SemanticError("unsupported collective")
    return MappingProxyType(outputs)
