"""Source contributions and actual storage lifetimes for v5 state versions."""
from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_left
import math

from vericcl.errors import SemanticError
from vericcl.semantics.ranges import DataRange, nonnegative


def _time(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or value < 0 or not math.isfinite(value):
        raise SemanticError("time must be finite and nonnegative")
    return float(value)


@dataclass(frozen=True)
class BufferRegion:
    rank: int
    buffer_id: str
    offset: int
    size: int
    kind: str = "data"

    def __post_init__(self):
        for field in ("rank", "offset", "size"):
            nonnegative(getattr(self, field), field)
        if not self.size or not isinstance(self.buffer_id, str) or not self.buffer_id:
            raise SemanticError("buffer region needs an identity and positive size")
        if self.kind not in {"data", "input", "output", "scratch"}:
            raise SemanticError("invalid buffer storage kind")

    def overlaps(self, other: BufferRegion) -> bool:
        return ((self.rank, self.buffer_id) == (other.rank, other.buffer_id)
                and max(self.offset, other.offset) < min(self.offset + self.size, other.offset + other.size))


@dataclass(frozen=True)
class StateVersion:
    state_id: str
    rank: int
    logical_start: int
    logical_end: int
    contributors: tuple[DataRange, ...]
    storage: BufferRegion
    ready_us: float

    def __post_init__(self):
        if not isinstance(self.state_id, str) or not self.state_id:
            raise SemanticError("state version needs a nonempty identity")
        for field in ("rank", "logical_start", "logical_end"):
            nonnegative(getattr(self, field), field)
        if self.logical_end <= self.logical_start:
            raise SemanticError("logical range must be nonempty")
        if not isinstance(self.storage, BufferRegion) or self.storage.rank != self.rank or self.storage.size != self.logical_end - self.logical_start:
            raise SemanticError("state storage and logical payload sizes must agree")
        pieces = tuple(self.contributors)
        if not pieces or not all(isinstance(piece, DataRange) for piece in pieces):
            raise SemanticError("state needs source contribution ranges")
        canonical = []
        for source in sorted({piece.source_rank for piece in pieces}):
            cursor = self.logical_start
            for piece in sorted(p for p in pieces if p.source_rank == source):
                if piece.start != cursor or piece.end > self.logical_end:
                    raise SemanticError("contribution has a duplicate, gap or different logical range")
                cursor = piece.end
            if cursor != self.logical_end:
                raise SemanticError("contribution does not cover the full logical range")
            canonical.append(DataRange(source, self.logical_start, self.logical_end))
        object.__setattr__(self, "contributors", tuple(canonical))
        object.__setattr__(self, "ready_us", _time(self.ready_us))


class VersionLedger:
    """Keep old state identities as evidence; validity follows storage writes."""
    def __init__(self, *, scratch_budgets: dict[int, int]):
        self.scratch_budgets = dict(scratch_budgets)
        for rank, budget in self.scratch_budgets.items():
            nonnegative(rank, "scratch rank")
            nonnegative(budget, "scratch budget")
        self._states = {}
        self._active = set()
        self._readers = {}
        self._buffer_kinds = {}
        self._regions = {}
        self._reader_end = {}
        self._scratch_live = {}

    @property
    def states(self):
        return tuple(self._states[key] for key in sorted(self._states))

    def state(self, state_id: str) -> StateVersion:
        if state_id not in self._states:
            raise SemanticError("unknown state version: " + state_id)
        return self._states[state_id]

    def is_active(self, state_id: str) -> bool:
        return state_id in self._active

    def readable(self, state_id: str, at_us: float) -> StateVersion:
        state = self.state(state_id)
        if state_id not in self._active:
            raise SemanticError("state version is no longer valid: " + state_id)
        if state.ready_us > _time(at_us):
            raise SemanticError("state is read before it is ready: " + state_id)
        return state

    def reserve_read(self, state_id: str, reader_id: str, *, start_us: float, end_us: float):
        self.readable(state_id, start_us)
        if _time(end_us) < _time(start_us):
            raise SemanticError("reader ends before it starts")
        if not isinstance(reader_id, str) or not reader_id or ("external", reader_id) in self._readers:
            raise SemanticError("reader identity must be nonempty and unique")
        self._readers[("external", reader_id)] = (state_id, float(end_us))
        self._reader_end[state_id] = max(self._reader_end.get(state_id, 0.), float(end_us))

    def _no_readers_after(self, state_id, at_us):
        if self._reader_end.get(state_id, 0.) > at_us:
            raise SemanticError("state has an unfinished reader: " + state_id)

    def _overlapping(self, region):
        # Active regions in one buffer are disjoint. At most one predecessor
        # can intersect the new start; all later intersections start before end.
        entries = self._regions.get((region.rank, region.buffer_id), [])
        first = bisect_left(entries, (region.offset, ''))
        if first and self._states[entries[first - 1][1]].storage.offset + self._states[entries[first - 1][1]].storage.size > region.offset:
            first -= 1
        last = bisect_left(entries, (region.offset + region.size, ''))
        return {state_id for _, state_id in entries[first:last]}

    def _remove_active(self, state_id):
        region = self._states[state_id].storage
        entries = self._regions[(region.rank, region.buffer_id)]
        entries.pop(bisect_left(entries, (region.offset, state_id)))
        self._active.remove(state_id)
        if region.kind == 'scratch':
            self._scratch_live[region.rank] -= region.size

    def _store(self, state, *, start_us, end_us, read_ids=(), initial=False):
        start_us, end_us = _time(start_us), _time(end_us)
        if end_us < start_us or state.ready_us != end_us:
            raise SemanticError("write interval and output ready time must agree")
        if state.state_id in self._states:
            raise SemanticError("state version identity already exists")
        reads = tuple(self.readable(key, start_us) for key in read_ids)
        region = state.storage
        key = region.rank, region.buffer_id
        if key in self._buffer_kinds and self._buffer_kinds[key] != region.kind:
            raise SemanticError("buffer alias cannot change its storage kind")
        for value in reads:
            if region.overlaps(value.storage) and region != value.storage:
                raise SemanticError("partial overlap with a read input requires separate storage")
        overwritten = self._overlapping(region)
        if initial and overwritten:
            raise SemanticError("initial states overlap storage")
        for key in overwritten:
            if self._states[key].ready_us > start_us:
                raise SemanticError("buffer still has an unfinished write: " + key)
            self._no_readers_after(key, start_us)
        if region.kind == "scratch":
            live = self._scratch_live.get(region.rank, 0) - sum(self._states[key].storage.size for key in overwritten)
            if region.rank not in self.scratch_budgets or live + region.size > self.scratch_budgets[region.rank]:
                raise SemanticError("scratch storage budget exceeded")
        # Commit only after all checks. Failed speculative operations leave the
        # active states, readers, identities and budgets unchanged.
        for index, value in enumerate(reads):
            self._readers[("operation", state.state_id, index)] = (value.state_id, end_us)
            self._reader_end[value.state_id] = max(self._reader_end.get(value.state_id, 0.), end_us)
        for key in overwritten:
            self._remove_active(key)
        self._states[state.state_id] = state
        self._active.add(state.state_id)
        entries = self._regions.setdefault((region.rank, region.buffer_id), [])
        entries.insert(bisect_left(entries, (region.offset, state.state_id)), (region.offset, state.state_id))
        if region.kind == 'scratch':
            self._scratch_live[region.rank] = self._scratch_live.get(region.rank, 0) + region.size
        self._buffer_kinds[(region.rank, region.buffer_id)] = region.kind
        return state

    def add_initial(self, state: StateVersion):
        return self._store(state, start_us=state.ready_us, end_us=state.ready_us, initial=True)

    def copy(self, source_id, *, state_id, destination, start_us, end_us):
        source = self.readable(source_id, start_us)
        result = StateVersion(state_id, destination.rank, source.logical_start, source.logical_end,
                              source.contributors, destination, end_us)
        return self._store(result, start_us=start_us, end_us=end_us, read_ids=(source_id,))

    def reduce(self, left_id, right_id, *, state_id, destination, start_us, end_us):
        left, right = self.readable(left_id, start_us), self.readable(right_id, start_us)
        if (left.logical_start, left.logical_end) != (right.logical_start, right.logical_end):
            raise SemanticError("reduction inputs must have the same logical range")
        if {p.source_rank for p in left.contributors} & {p.source_rank for p in right.contributors}:
            raise SemanticError("reduction contribution is counted more than once")
        if destination.rank not in {left.rank, right.rank}:
            raise SemanticError("reduction destination must own an input")
        result = StateVersion(state_id, destination.rank, left.logical_start, left.logical_end,
                              left.contributors + right.contributors, destination, end_us)
        return self._store(result, start_us=start_us, end_us=end_us, read_ids=(left_id, right_id))

    def release(self, state_id, *, at_us):
        self.readable(state_id, at_us)
        self._no_readers_after(state_id, at_us)
        self._remove_active(state_id)
