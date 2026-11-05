"""Immutable source byte ranges, independent of the current slice partition."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from vericcl.errors import SemanticError


def nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SemanticError(name + " must be a nonnegative integer")
    return value


@dataclass(frozen=True, order=True)
class DataRange:
    source_rank: int
    start: int
    end: int

    def __post_init__(self):
        for field in ("source_rank", "start", "end"):
            nonnegative(getattr(self, field), field)
        if self.end <= self.start:
            raise SemanticError("source data range must be nonempty")

    def intersects(self, other: DataRange) -> bool:
        return self.source_rank == other.source_rank and max(self.start, other.start) < min(self.end, other.end)


def reference_slice_ids(data: DataRange, *, input_bytes: int, reference_slice_bytes: int) -> tuple[int, ...]:
    for value, name in ((input_bytes, "input_bytes"), (reference_slice_bytes, "reference_slice_bytes")):
        nonnegative(value, name)
        if value == 0:
            raise SemanticError(name + " must be positive")
    if input_bytes % reference_slice_bytes or data.end > input_bytes:
        raise SemanticError("range is outside a legal reference partition")
    count = input_bytes // reference_slice_bytes
    return tuple(data.source_rank * count + index for index in range(data.start // reference_slice_bytes,
                                                                    (data.end - 1) // reference_slice_bytes + 1))


@dataclass(frozen=True)
class EdgeBan:
    data: DataRange
    src_rank: int
    dst_rank: int
    stage_id: int
    original_stage_id: int


@dataclass(frozen=True)
class ConstraintSet:
    bans: tuple[EdgeBan, ...]
    rank_count: int
    fixed_stage_count: int | None = None

    def allows(self, contributors, src_rank: int, dst_rank: int, stage_id: int) -> bool:
        for value, name in ((src_rank, "src_rank"), (dst_rank, "dst_rank"), (stage_id, "stage_id")):
            nonnegative(value, name)
        if src_rank >= self.rank_count or dst_rank >= self.rank_count or src_rank == dst_rank:
            return False
        if self.fixed_stage_count is not None and stage_id >= self.fixed_stage_count:
            return False
        payload = tuple(contributors)
        if not payload or not all(isinstance(item, DataRange) for item in payload):
            raise SemanticError("constraint check needs nonempty source ranges")
        return not any((ban.src_rank, ban.dst_rank, ban.stage_id) == (src_rank, dst_rank, stage_id)
                       and any(ban.data.intersects(piece) for piece in payload) for ban in self.bans)

    def remap_stages(self, mapping: Mapping[int, tuple[int, ...]], *, new_stage_count: int) -> ConstraintSet:
        """Carry bans through a caller-validated logical-stage correspondence.

        This checks constraint coverage and cardinality. The plan interface
        checker separately validates that each correspondence preserves meaning.
        """
        nonnegative(new_stage_count, "new_stage_count")
        if not new_stage_count or (self.fixed_stage_count is not None and self.fixed_stage_count != new_stage_count):
            raise SemanticError("user-fixed stage count cannot change")
        for ban in self.bans:
            if ban.stage_id not in mapping or not mapping[ban.stage_id]:
                raise SemanticError("stage mapping does not cover a restricted logical stage")
        for old, targets in mapping.items():
            nonnegative(old, "old stage")
            if not targets or len(set(targets)) != len(targets):
                raise SemanticError("stage mapping targets must be nonempty and unique")
            for target in targets:
                nonnegative(target, "target stage")
                if target >= new_stage_count:
                    raise SemanticError("stage mapping target is outside new stage count")
        return ConstraintSet(tuple(replace(ban, stage_id=stage) for ban in self.bans for stage in mapping[ban.stage_id]),
                             self.rank_count, self.fixed_stage_count)


def bind_constraints(inputs) -> ConstraintSet:
    hyper = inputs.hyperparameters
    reference = hyper.reference_slice_size_bytes or hyper.slice_size_bytes
    if inputs.atom_constraints.forbidden_transfers and reference is None:
        raise SemanticError("reference slice size is required to bind hard constraints")
    count = None if reference is None else hyper.total_size_bytes // reference
    bans = []
    for item in inputs.atom_constraints.forbidden_transfers:
        rank, index = divmod(item.slice_id, count)
        if rank >= inputs.rank_count:
            raise SemanticError("constraint source slice is outside the reference partition")
        bans.append(EdgeBan(DataRange(rank, index * reference, (index + 1) * reference),
                            item.src_rank, item.dst_rank, item.stage_id, item.stage_id))
    return ConstraintSet(tuple(bans), inputs.rank_count, inputs.atom_constraints.stage_num)
