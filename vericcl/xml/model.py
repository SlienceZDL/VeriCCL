from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Tuple, Union

from vericcl.errors import SemanticError
from vericcl.semantics.collective import OutputSlot


def _integer(value: object, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SemanticError("{} must be an integer".format(field))
    if value < minimum:
        raise SemanticError("{} must be at least {}".format(field, minimum))
    return value


def _time(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticError("{} must be a number".format(field))
    normalized = float(value)
    if math.isnan(normalized) or normalized < 0.0:
        raise SemanticError("{} must be non-negative".format(field))
    return normalized


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SemanticError("{} must be a non-empty string".format(field))
    return value


@dataclass(frozen=True, order=True)
class RawValue:
    slice_id: int

    def __post_init__(self) -> None:
        _integer(self.slice_id, "raw_value.slice_id")


@dataclass(frozen=True, order=True)
class AggregateValue:
    logical_slice_index: int
    contributors: frozenset[int]
    state_version: int

    def __post_init__(self) -> None:
        _integer(
            self.logical_slice_index,
            "aggregate_value.logical_slice_index",
        )
        contributors = frozenset(self.contributors)
        if len(contributors) < 2:
            raise SemanticError(
                "aggregate_value.contributors must contain at least two slices"
            )
        for contributor in contributors:
            _integer(contributor, "aggregate_value.contributors")
        object.__setattr__(self, "contributors", contributors)
        _integer(self.state_version, "aggregate_value.state_version")


ValueKey = Union[RawValue, AggregateValue]


@dataclass(frozen=True)
class PhysicalRef:
    rank: int
    buffer: str
    offset: int
    valid_from: float
    valid_until: float

    def __post_init__(self) -> None:
        _integer(self.rank, "physical_ref.rank")
        if self.buffer not in {"i", "o", "s"}:
            raise SemanticError("physical_ref.buffer must be i, o, or s")
        _integer(self.offset, "physical_ref.offset")
        object.__setattr__(
            self,
            "valid_from",
            _time(self.valid_from, "physical_ref.valid_from"),
        )
        object.__setattr__(
            self,
            "valid_until",
            _time(self.valid_until, "physical_ref.valid_until"),
        )
        if self.valid_from > self.valid_until:
            raise SemanticError(
                "physical_ref.valid_from must not exceed valid_until"
            )

    @property
    def buffer_offset(self) -> Tuple[str, int]:
        return self.buffer, self.offset


@dataclass(frozen=True)
class LocalCopy:
    copy_id: str
    rank: int
    src_ref: PhysicalRef
    dst_ref: PhysicalRef
    predecessor_state_id: str
    st_time: float
    ed_time: float
    reason: str

    def __post_init__(self) -> None:
        _identifier(self.copy_id, "local_copy.copy_id")
        _integer(self.rank, "local_copy.rank")
        if self.src_ref.rank != self.rank or self.dst_ref.rank != self.rank:
            raise SemanticError("local copy references must use the copy rank")
        _identifier(
            self.predecessor_state_id,
            "local_copy.predecessor_state_id",
        )
        object.__setattr__(
            self,
            "st_time",
            _time(self.st_time, "local_copy.st_time"),
        )
        object.__setattr__(
            self,
            "ed_time",
            _time(self.ed_time, "local_copy.ed_time"),
        )
        if self.st_time > self.ed_time:
            raise SemanticError("local copy start must not exceed end")
        _identifier(self.reason, "local_copy.reason")


def _mapping(value: Mapping, field: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise SemanticError("{} must be a mapping".format(field))
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class BufferPlan:
    value_locations: Mapping[ValueKey, Tuple[PhysicalRef, ...]]
    aliases: Tuple[Tuple[PhysicalRef, PhysicalRef], ...]
    local_copies: Tuple[LocalCopy, ...]
    i_chunks: Mapping[int, int]
    o_chunks: Mapping[int, int]
    s_chunks: Mapping[int, int]
    slice_count: int
    initial_refs: Mapping[int, PhysicalRef]
    final_output_refs: Mapping[OutputSlot, PhysicalRef]
    final_values: Mapping[OutputSlot, ValueKey]
    final_value_refs: Mapping[OutputSlot, PhysicalRef]
    transfer_src_refs: Mapping[str, PhysicalRef]
    transfer_dst_refs: Mapping[str, PhysicalRef]
    transfer_accumulator_refs: Mapping[str, PhysicalRef]
    transfer_input_values: Mapping[str, ValueKey]
    transfer_accumulator_values: Mapping[str, ValueKey]
    transfer_output_values: Mapping[str, ValueKey]
    transfer_effective_times: Mapping[str, Tuple[float, float]]

    def __post_init__(self) -> None:
        locations = {
            value: tuple(refs) for value, refs in self.value_locations.items()
        }
        for value, refs in locations.items():
            if not isinstance(value, (RawValue, AggregateValue)):
                raise SemanticError("buffer plan contains an invalid value key")
            if not all(isinstance(ref, PhysicalRef) for ref in refs):
                raise SemanticError(
                    "buffer plan value locations must contain PhysicalRef values"
                )
        object.__setattr__(
            self,
            "value_locations",
            MappingProxyType(locations),
        )
        aliases = tuple(self.aliases)
        if not all(
            isinstance(pair, tuple)
            and len(pair) == 2
            and all(isinstance(ref, PhysicalRef) for ref in pair)
            for pair in aliases
        ):
            raise SemanticError("buffer plan aliases must contain reference pairs")
        object.__setattr__(self, "aliases", aliases)
        copies = tuple(self.local_copies)
        if not all(isinstance(copy, LocalCopy) for copy in copies):
            raise SemanticError("buffer plan local_copies are invalid")
        object.__setattr__(self, "local_copies", copies)
        _integer(self.slice_count, "buffer_plan.slice_count", minimum=1)
        for field in (
            "i_chunks",
            "o_chunks",
            "s_chunks",
            "initial_refs",
            "final_output_refs",
            "final_values",
            "final_value_refs",
            "transfer_src_refs",
            "transfer_dst_refs",
            "transfer_accumulator_refs",
            "transfer_input_values",
            "transfer_accumulator_values",
            "transfer_output_values",
            "transfer_effective_times",
        ):
            object.__setattr__(self, field, _mapping(getattr(self, field), field))

    def final_ref(
        self,
        rank: int,
        source_rank: int,
        logical_slice_index: int,
    ) -> PhysicalRef:
        _integer(rank, "final_ref.rank")
        _integer(source_rank, "final_ref.source_rank")
        _integer(logical_slice_index, "final_ref.logical_slice_index")
        contributor = source_rank * self.slice_count + logical_slice_index
        matches = [
            self.final_output_refs[slot]
            for slot, value in self.final_values.items()
            if slot.rank == rank
            and (
                isinstance(value, RawValue)
                and value.slice_id == contributor
                or isinstance(value, AggregateValue)
                and contributor in value.contributors
            )
        ]
        if len(matches) != 1:
            raise SemanticError("final value does not map to exactly one output")
        return matches[0]
