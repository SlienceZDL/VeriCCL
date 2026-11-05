"""Executable decisions expressed using source ranges and storage versions."""
from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right

from vericcl.errors import SemanticError
from vericcl.execution.budget import BudgetExhausted
from vericcl.semantics.collective import CollectiveKind
from vericcl.semantics.ranges import bind_constraints, nonnegative
from vericcl.semantics.versions import BufferRegion, StateVersion, VersionLedger, _time


@dataclass(frozen=True)
class OutputBinding:
    rank: int
    offset_bytes: int
    state_id: str

    def __post_init__(self):
        nonnegative(self.rank, "output rank")
        nonnegative(self.offset_bytes, "output offset")
        if not isinstance(self.state_id, str) or not self.state_id:
            raise SemanticError("output needs a state identity")


@dataclass(frozen=True)
class VersionOperation:
    operation_id: str
    kind: str
    input_states: tuple[str, ...]
    output_state: str | None
    destination: BufferRegion | None
    start_us: float
    end_us: float
    stage_id: int
    dependencies: tuple[str, ...] = ()
    participants: tuple[int, ...] = ()

    def __post_init__(self):
        if not isinstance(self.operation_id, str) or not self.operation_id:
            raise SemanticError("operation needs a stable identity")
        if self.kind not in {"copy", "reduce", "release"}:
            raise SemanticError("unknown version operation kind")
        count = 2 if self.kind == "reduce" else 1
        if len(self.input_states) != count or len(set(self.input_states)) != count:
            raise SemanticError("operation has invalid input state identities")
        nonnegative(self.stage_id, "operation stage")
        if _time(self.end_us) < _time(self.start_us):
            raise SemanticError("operation ends before it starts")
        if self.kind == "release":
            if self.output_state is not None or self.destination is not None:
                raise SemanticError("release does not create a state")
        elif not self.output_state or not isinstance(self.destination, BufferRegion):
            raise SemanticError("write operation needs output state and storage")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise SemanticError("duplicate operation dependency")
        if len(set(self.participants)) != len(self.participants):
            raise SemanticError("duplicate participant rank")
        for rank in self.participants:
            nonnegative(rank, "participant rank")


@dataclass(frozen=True)
class VersionSchedule:
    rank_count: int
    input_bytes: int
    slice_size_bytes: int
    stage_count: int
    initial_states: tuple[StateVersion, ...]
    operations: tuple[VersionOperation, ...]
    outputs: tuple[OutputBinding, ...]

    def __post_init__(self):
        for name in ("rank_count", "input_bytes", "slice_size_bytes", "stage_count"):
            nonnegative(getattr(self, name), name)
            if not getattr(self, name):
                raise SemanticError(name + " must be positive")
        if self.input_bytes % self.slice_size_bytes:
            raise SemanticError("schedule slice size must divide input size")
        ids = [op.operation_id for op in self.operations]
        if len(set(ids)) != len(ids):
            raise SemanticError("operation identities must be unique")
        if any(op.stage_id >= self.stage_count for op in self.operations):
            raise SemanticError("operation stage exceeds logical stage count")


def _check_interval_cover(intervals, start, end, label):
    cursor = start
    for left, right in sorted(intervals):
        if left != cursor or right <= left or right > end:
            raise SemanticError(label + " has a gap, duplicate or out-of-range interval")
        cursor = right
    if cursor != end:
        raise SemanticError(label + " is incomplete")


def _initial_states(schedule, inputs, ledger, lease=None):
    source_ranks = ({inputs.collective.root} if inputs.collective.kind is CollectiveKind.BROADCAST
                    else set(range(schedule.rank_count)))
    covered = {rank: [] for rank in source_ranks}
    for state in schedule.initial_states:
        if lease is not None:
            lease.check()
        if (state.rank not in source_ranks or state.ready_us != 0 or len(state.contributors) != 1
                or state.contributors[0].source_rank != state.rank):
            raise SemanticError("initial state must contain only its rank's original input")
        if state.logical_end - state.logical_start != schedule.slice_size_bytes or state.logical_start % schedule.slice_size_bytes:
            raise SemanticError("initial state does not match the resolved slice partition")
        covered[state.rank].append((state.logical_start, state.logical_end))
        ledger.add_initial(state)
    for rank, intervals in covered.items():
        _check_interval_cover(intervals, 0, schedule.input_bytes, "initial input for rank " + str(rank))


def _expected_segments(spec, ranks, size):
    """Return (rank, output start/end, original logical start, source ranks)."""
    kind = spec.kind
    all_sources = frozenset(range(ranks))
    if kind in {CollectiveKind.BROADCAST, CollectiveKind.REDUCE, CollectiveKind.ALL_REDUCE}:
        destinations = (spec.root,) if kind is CollectiveKind.REDUCE else range(ranks)
        sources = frozenset({spec.root}) if kind is CollectiveKind.BROADCAST else all_sources
        return tuple((rank, 0, size, 0, sources) for rank in destinations)
    if kind is CollectiveKind.ALL_GATHER:
        return tuple((rank, source * size, (source + 1) * size, 0, frozenset({source}))
                     for rank in range(ranks) for source in range(ranks))
    if kind in {CollectiveKind.ALL_TO_ALL, CollectiveKind.REDUCE_SCATTER}:
        if size % ranks:
            raise SemanticError("partitioned collective input must divide by rank count")
        part = size // ranks
        if kind is CollectiveKind.REDUCE_SCATTER:
            return tuple((rank, 0, part, rank * part, all_sources) for rank in range(ranks))
        return tuple((rank, source * part, (source + 1) * part, rank * part, frozenset({source}))
                     for rank in range(ranks) for source in range(ranks))
    raise SemanticError("unsupported external output semantics")


def check_version_outputs(spec, ranks, input_bytes, outputs, ledger, *, at_us, lease=None):
    expected = _expected_segments(spec, ranks, input_bytes)
    by_rank = {}
    for index, segment in enumerate(expected):
        by_rank.setdefault(segment[0], []).append((segment[1], index))
    coverage = {index: [] for index in range(len(expected))}
    used_states = set()
    for output in outputs:
        if lease is not None:
            lease.check()
        state = ledger.readable(output.state_id, at_us)
        if output.state_id in used_states or state.rank != output.rank:
            raise SemanticError("duplicate output state or wrong output rank")
        used_states.add(output.state_id)
        out_end = output.offset_bytes + state.logical_end - state.logical_start
        segments = by_rank.get(output.rank, [])
        position = bisect_right(segments, (output.offset_bytes, len(expected))) - 1
        if position < 0:
            raise SemanticError("output range does not belong to an expected result segment")
        index = segments[position][1]
        _, start, end, logical_start, sources = expected[index]
        if out_end > end:
            raise SemanticError("output range does not belong to an expected result segment")
        if (state.logical_start != logical_start + output.offset_bytes - start
                or frozenset(p.source_rank for p in state.contributors) != sources):
            raise SemanticError("output has missing, repeated or wrong original contributions")
        coverage[index].append((output.offset_bytes, out_end))
    totals = {rank: 0 for rank in range(ranks)}
    for index, (rank, start, end, _, _) in enumerate(expected):
        if lease is not None:
            lease.check()
        _check_interval_cover(coverage[index], start, end, "output for rank " + str(rank))
        totals[rank] += end - start
    return totals


def replay_version_schedule(schedule: VersionSchedule, inputs, *, lease=None):
    if lease is not None:
        lease.check()
    if (schedule.rank_count != inputs.rank_count or schedule.input_bytes != inputs.hyperparameters.total_size_bytes
            or schedule.slice_size_bytes != inputs.hyperparameters.slice_size_bytes):
        raise SemanticError("schedule dimensions differ from the resolved partition")
    if inputs.atom_constraints.stage_num is not None and schedule.stage_count != inputs.atom_constraints.stage_num:
        raise SemanticError("schedule does not preserve fixed stage count")
    budget = inputs.request_config["runtime"]["scratch_bytes_per_rank"]
    ledger = VersionLedger(scratch_budgets={rank: budget for rank in range(schedule.rank_count)})
    _initial_states(schedule, inputs, ledger, lease)
    constraints = bind_constraints(inputs)
    links = {(edge["src"], edge["dst"]) for edge in inputs.resolved_topology["directed_links"]}
    finished = {}
    for op in sorted(schedule.operations, key=lambda item: (item.start_us, item.end_us, item.operation_id)):
        if lease is not None:
            lease.check()
        try:
            for dependency in op.dependencies:
                if dependency not in finished or finished[dependency] > op.start_us:
                    raise SemanticError("explicit dependency is not complete")
            states = tuple(ledger.readable(key, op.start_us) for key in op.input_states)
            if op.kind == "release":
                ledger.release(states[0].state_id, at_us=op.start_us)
            else:
                if op.destination.rank >= schedule.rank_count:
                    raise SemanticError("destination rank is outside the group")
                for state in states:
                    if op.participants and (state.rank not in op.participants or op.destination.rank not in op.participants):
                        raise SemanticError("operation uses a GPU outside its explicit participant group")
                    if state.rank != op.destination.rank:
                        if (state.rank, op.destination.rank) not in links:
                            raise SemanticError("operation uses an absent directed link")
                        if not constraints.allows(state.contributors, state.rank, op.destination.rank, op.stage_id):
                            raise SemanticError("operation violates a source-range directed-edge constraint")
                method = ledger.reduce if op.kind == "reduce" else ledger.copy
                method(*op.input_states, state_id=op.output_state, destination=op.destination,
                       start_us=op.start_us, end_us=op.end_us)
            finished[op.operation_id] = op.end_us
        except BudgetExhausted:
            raise
        except SemanticError as error:
            raise SemanticError("operation " + op.operation_id + ": " + str(error)) from error
    at_us = max(finished.values(), default=0.0)
    output_bytes = check_version_outputs(inputs.collective, schedule.rank_count, schedule.input_bytes,
                                        schedule.outputs, ledger, at_us=at_us, lease=lease)
    return ledger, {"output_bytes": output_bytes, "state_versions": len(ledger.states),
                    "operations": len(schedule.operations), "scope": "data semantics and declared storage lifetimes"}
