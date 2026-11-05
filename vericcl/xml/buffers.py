from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from vericcl.errors import SemanticError
from vericcl.input.models import ResolvedInput
from vericcl.semantics.atom import Schedule, Transfer
from vericcl.semantics.collective import (
    CollectiveKind,
    OutputSlot,
    required_outputs,
)
from vericcl.semantics.slice import logical_slice_index
from vericcl.xml.model import (
    AggregateValue,
    BufferPlan,
    LocalCopy,
    PhysicalRef,
    RawValue,
    ValueKey,
)


Token = Tuple[int, str, object]


@dataclass
class _Occurrence:
    occurrence_id: str
    value: ValueKey
    token: Token
    valid_from: float
    last_use: float
    closed_at: Optional[float] = None
    final: bool = False
    immutable: bool = False

    def touch(self, time: float) -> None:
        self.last_use = max(self.last_use, time)

    def close(self, time: float) -> None:
        self.touch(time)
        self.closed_at = time

    @property
    def valid_until(self) -> float:
        if self.final or self.immutable:
            return math.inf
        if self.closed_at is not None:
            return self.closed_at
        return self.last_use


@dataclass
class _State:
    value: ValueKey
    rank: int
    logical: int
    contributors: frozenset[int]
    occurrence: _Occurrence
    ready_time: float
    producer_id: str


@dataclass
class _CopyDraft:
    copy_id: str
    rank: int
    src: _Occurrence
    dst: _Occurrence
    predecessor_state_id: str
    st_time: float
    ed_time: float
    reason: str


def _input_token(rank: int, offset: int) -> Token:
    return rank, "i", offset


def _output_token(rank: int, offset: int) -> Token:
    return rank, "o", offset


def _stable_transfer_order(schedule: Schedule) -> Tuple[Transfer, ...]:
    by_id = {transfer.transfer_id: transfer for transfer in schedule.transfers}
    raw_semantic = schedule.metadata.get("semantic_predecessors", {})
    if raw_semantic is None:
        raw_semantic = {}
    if not isinstance(raw_semantic, Mapping):
        raise SemanticError("semantic_predecessors metadata must be a mapping")
    dependencies = {}
    for transfer in schedule.transfers:
        semantic = raw_semantic.get(transfer.transfer_id, ())
        try:
            semantic_ids = frozenset(semantic)
        except TypeError as error:
            raise SemanticError(
                "semantic predecessor IDs must be iterable"
            ) from error
        dependencies[transfer.transfer_id] = (
            frozenset(transfer.predecessor_ids) | semantic_ids
        )
        if not dependencies[transfer.transfer_id] <= set(by_id):
            raise SemanticError("buffer planning dependency is missing")
    remaining = dict(dependencies)
    completed = set()
    ordered = []
    while remaining:
        ready = [
            by_id[transfer_id]
            for transfer_id, predecessors in remaining.items()
            if predecessors <= completed
        ]
        if not ready:
            raise SemanticError("buffer planning dependencies contain a cycle")
        selected = min(
            ready,
            key=lambda transfer: (
                transfer.st_time,
                transfer.ed_time,
                transfer.stage_id,
                transfer.src_rank,
                transfer.dst_rank,
                transfer.channel,
                transfer.transfer_id,
            ),
        )
        ordered.append(selected)
        completed.add(selected.transfer_id)
        del remaining[selected.transfer_id]
    return tuple(ordered)


def _alias_pairs(
    kind: CollectiveKind,
    rank_count: int,
    slice_count: int,
    root: Optional[int],
) -> Tuple[Tuple[int, int, int], ...]:
    pairs = []
    if kind in {CollectiveKind.BROADCAST, CollectiveKind.ALL_REDUCE}:
        for rank in range(rank_count):
            for logical in range(slice_count):
                pairs.append((rank, logical, logical))
    elif kind is CollectiveKind.REDUCE:
        if root is None:
            raise SemanticError("Reduce buffer planning requires a root")
        for logical in range(slice_count):
            pairs.append((root, logical, logical))
    elif kind is CollectiveKind.ALL_GATHER:
        for rank in range(rank_count):
            for logical in range(slice_count):
                pairs.append(
                    (rank, logical, rank * slice_count + logical)
                )
    elif kind is CollectiveKind.REDUCE_SCATTER:
        quotient = slice_count // rank_count
        for rank in range(rank_count):
            for offset in range(quotient):
                pairs.append(
                    (rank, rank * quotient + offset, offset)
                )
    elif kind is CollectiveKind.ALL_TO_ALL:
        for rank in range(rank_count):
            for offset in range(slice_count):
                pairs.append((rank, offset, offset))
    return tuple(pairs)


def _output_chunks(
    kind: CollectiveKind,
    rank_count: int,
    slice_count: int,
) -> int:
    if kind is CollectiveKind.ALL_GATHER:
        return rank_count * slice_count
    if kind is CollectiveKind.REDUCE_SCATTER:
        return slice_count // rank_count
    return slice_count


def _value_for_contributors(
    contributors: frozenset[int],
    logical: int,
    version: int,
) -> ValueKey:
    if len(contributors) == 1:
        return RawValue(next(iter(contributors)))
    return AggregateValue(logical, contributors, version)


def _scratch_intervals(
    occurrences: Iterable[_Occurrence],
) -> Mapping[Token, Tuple[float, float]]:
    intervals: Dict[Token, Tuple[float, float]] = {}
    for occurrence in occurrences:
        if occurrence.token[1] != "s":
            continue
        start, end = intervals.get(
            occurrence.token,
            (occurrence.valid_from, occurrence.valid_until),
        )
        intervals[occurrence.token] = (
            min(start, occurrence.valid_from),
            max(end, occurrence.valid_until),
        )
    return intervals


def _allocate_scratch(
    occurrences: Iterable[_Occurrence],
    rank_count: int,
) -> Tuple[Mapping[Token, int], Mapping[int, int]]:
    intervals = _scratch_intervals(occurrences)
    offsets = {}
    counts = {rank: 0 for rank in range(rank_count)}
    by_rank = defaultdict(list)
    for token, interval in intervals.items():
        by_rank[token[0]].append((interval[0], interval[1], str(token[2]), token))
    for rank in range(rank_count):
        slot_ends: List[float] = []
        for start, end, _, token in sorted(by_rank[rank]):
            offset = next(
                (
                    candidate
                    for candidate, slot_end in enumerate(slot_ends)
                    if slot_end <= start
                ),
                len(slot_ends),
            )
            if offset == len(slot_ends):
                slot_ends.append(end)
            else:
                slot_ends[offset] = end
            offsets[token] = offset
        counts[rank] = len(slot_ends)
    return offsets, counts


def build_buffer_plan(
    schedule: Schedule,
    inputs: ResolvedInput,
) -> BufferPlan:
    if not isinstance(schedule, Schedule):
        raise SemanticError("schedule must be a Schedule")
    if not isinstance(inputs, ResolvedInput):
        raise SemanticError("inputs must be a ResolvedInput")
    if schedule.rank_count != inputs.rank_count:
        raise SemanticError("schedule and input rank counts differ")
    if schedule.slice_count != inputs.hyperparameters.slice_count:
        raise SemanticError("schedule and input slice counts differ")
    if schedule.slice_size_bytes != inputs.hyperparameters.slice_size_bytes:
        raise SemanticError("schedule and input slice sizes differ")

    rank_count = schedule.rank_count
    slice_count = schedule.slice_count
    spec = inputs.collective
    expected_outputs = required_outputs(spec, rank_count, slice_count)
    transfers_by_id = {
        transfer.transfer_id: transfer for transfer in schedule.transfers
    }
    i_chunks = {rank: slice_count for rank in range(rank_count)}
    output_chunks = _output_chunks(spec.kind, rank_count, slice_count)
    o_chunks = {rank: output_chunks for rank in range(rank_count)}

    alias_offsets = (
        _alias_pairs(
            spec.kind,
            rank_count,
            slice_count,
            spec.root,
        )
        if spec.inplace
        else ()
    )
    input_to_output = {
        _input_token(rank, input_offset): _output_token(rank, output_offset)
        for rank, input_offset, output_offset in alias_offsets
    }
    alias_tokens = {
        **input_to_output,
        **{output: input_ for input_, output in input_to_output.items()},
    }

    occurrences: List[_Occurrence] = []
    occurrence_index = 0
    scratch_index = 0
    copies: List[_CopyDraft] = []
    states_by_position: Dict[Tuple[int, int], List[_State]] = defaultdict(list)
    initial_occurrences: Dict[int, _Occurrence] = {}
    transfer_src_occurrences = {}
    transfer_dst_occurrences = {}
    transfer_accumulator_occurrences = {}
    transfer_input_values = {}
    transfer_accumulator_values = {}
    transfer_output_values = {}
    transfer_effective_times = {}
    aggregate_versions = defaultdict(int)

    def new_occurrence(
        value: ValueKey,
        token: Token,
        start: float,
        *,
        immutable: bool = False,
        final: bool = False,
    ) -> _Occurrence:
        nonlocal occurrence_index
        occurrence = _Occurrence(
            occurrence_id="location-{:08d}".format(occurrence_index),
            value=value,
            token=token,
            valid_from=start,
            last_use=start,
            immutable=immutable,
            final=final,
        )
        occurrence_index += 1
        occurrences.append(occurrence)
        return occurrence

    def new_scratch_token(rank: int, label: str) -> Token:
        nonlocal scratch_index
        token = rank, "s", "{:08d}-{}".format(scratch_index, label)
        scratch_index += 1
        return token

    def copy_state(
        state: _State,
        target: Token,
        time: float,
        reason: str,
        *,
        close_source: bool = False,
    ) -> _State:
        if state.occurrence.token == target:
            return state
        state.occurrence.touch(time)
        destination = new_occurrence(state.value, target, time)
        copies.append(
            _CopyDraft(
                copy_id="copy-{:08d}".format(len(copies)),
                rank=state.rank,
                src=state.occurrence,
                dst=destination,
                predecessor_state_id=state.producer_id,
                st_time=time,
                ed_time=time,
                reason=reason,
            )
        )
        if close_source:
            state.occurrence.close(time)
            initial = (
                initial_occurrences.get(state.value.slice_id)
                if isinstance(state.value, RawValue)
                else None
            )
            if initial is not None and initial.token[0] == state.rank:
                initial.close(time)
        return _State(
            value=state.value,
            rank=state.rank,
            logical=state.logical,
            contributors=state.contributors,
            occurrence=destination,
            ready_time=time,
            producer_id=state.producer_id,
        )

    for rank in range(rank_count):
        for logical in range(slice_count):
            slice_id = rank * slice_count + logical
            value = RawValue(slice_id)
            input_occurrence = new_occurrence(
                value,
                _input_token(rank, logical),
                0.0,
                immutable=not spec.inplace,
            )
            initial_occurrences[slice_id] = input_occurrence
            effective = input_occurrence
            alias = input_to_output.get(input_occurrence.token)
            if alias is not None and spec.kind is not CollectiveKind.ALL_TO_ALL:
                effective = new_occurrence(value, alias, 0.0)
            states_by_position[(rank, logical)].append(
                _State(
                    value=value,
                    rank=rank,
                    logical=logical,
                    contributors=frozenset({slice_id}),
                    occurrence=effective,
                    ready_time=0.0,
                    producer_id="initial-slice-{}".format(slice_id),
                )
            )

    raw_last_reads = defaultdict(float)
    for transfer in schedule.transfers:
        if len(transfer.member_slice_ids) == 1:
            member = next(iter(transfer.member_slice_ids))
            if member // slice_count == transfer.src_rank:
                raw_last_reads[member] = max(
                    raw_last_reads[member],
                    transfer.ed_time,
                )

    def find_state(
        rank: int,
        logical: int,
        contributors: frozenset[int],
    ) -> _State:
        matches = [
            state
            for state in states_by_position[(rank, logical)]
            if state.contributors == contributors
        ]
        if not matches:
            raise SemanticError(
                "transfer source value is unavailable at the source rank"
            )
        return max(
            matches,
            key=lambda state: (state.ready_time, state.producer_id),
        )

    def replace_state(old: _State, new: _State) -> None:
        states = states_by_position[(old.rank, old.logical)]
        states.remove(old)
        states.append(new)

    def final_slot(
        rank: int,
        contributors: frozenset[int],
    ) -> Optional[OutputSlot]:
        matches = [
            slot
            for slot, expected in expected_outputs.items()
            if slot.rank == rank and expected == contributors
        ]
        if len(matches) > 1:
            raise SemanticError("value maps to multiple final output slots")
        return matches[0] if matches else None

    for transfer in _stable_transfer_order(schedule):
        contributors = frozenset(transfer.member_slice_ids)
        logical_values = {
            logical_slice_index(member, slice_count)
            for member in contributors
        }
        if len(logical_values) != 1:
            raise SemanticError(
                "one physical transfer contains different logical slices"
            )
        logical = next(iter(logical_values))
        source = find_state(transfer.src_rank, logical, contributors)
        duration = transfer.ed_time - transfer.st_time

        if transfer.kind == "SEND":
            effective_start = max(transfer.st_time, source.ready_time)
            effective_end = effective_start + duration
            source.occurrence.touch(effective_end)
            transfer_src_occurrences[transfer.transfer_id] = source.occurrence
            transfer_input_values[transfer.transfer_id] = source.value
            transfer_effective_times[transfer.transfer_id] = (
                effective_start,
                effective_end,
            )
            slot = final_slot(transfer.dst_rank, contributors)
            target = (
                _output_token(transfer.dst_rank, slot.offset)
                if slot is not None
                else new_scratch_token(
                    transfer.dst_rank,
                    "recv-{}".format(transfer.transfer_id),
                )
            )
            if (
                spec.inplace
                and spec.kind is CollectiveKind.ALL_TO_ALL
                and target[1] == "o"
            ):
                input_offset = target[2]
                raw_id = transfer.dst_rank * slice_count + int(input_offset)
                if raw_last_reads[raw_id] > effective_start:
                    raw_state = find_state(
                        transfer.dst_rank,
                        int(input_offset),
                        frozenset({raw_id}),
                    )
                    if raw_state.occurrence.token[1] == "i":
                        preserved = copy_state(
                            raw_state,
                            new_scratch_token(
                                transfer.dst_rank,
                                "preserve-{}".format(raw_id),
                            ),
                            effective_start,
                            "preserve live in-place input",
                            close_source=True,
                        )
                        for prior_id, prior_value in tuple(
                            transfer_input_values.items()
                        ):
                            prior_transfer = transfers_by_id[prior_id]
                            if (
                                prior_value == raw_state.value
                                and prior_transfer.src_rank == transfer.dst_rank
                                and transfer_effective_times[prior_id][1]
                                > effective_start
                            ):
                                preserved.occurrence.touch(
                                    transfer_effective_times[prior_id][1]
                                )
                                transfer_src_occurrences[
                                    prior_id
                                ] = preserved.occurrence
                        replace_state(raw_state, preserved)
            destination = new_occurrence(
                source.value,
                target,
                effective_end,
            )
            transfer_dst_occurrences[transfer.transfer_id] = destination
            transfer_output_values[transfer.transfer_id] = source.value
            states = states_by_position[(transfer.dst_rank, logical)]
            states[:] = [
                state
                for state in states
                if state.contributors != contributors
            ]
            states.append(
                _State(
                    value=source.value,
                    rank=transfer.dst_rank,
                    logical=logical,
                    contributors=contributors,
                    occurrence=destination,
                    ready_time=effective_end,
                    producer_id=transfer.transfer_id,
                )
            )
            continue

        candidates = [
            state
            for state in states_by_position[(transfer.dst_rank, logical)]
            if state.contributors.isdisjoint(contributors)
        ]
        if not candidates:
            raise SemanticError("REDUCE destination accumulator is unavailable")
        local_raw = transfer.dst_rank * slice_count + logical
        accumulator = max(
            candidates,
            key=lambda state: (
                len(state.contributors),
                local_raw in state.contributors,
                state.ready_time,
                state.producer_id,
            ),
        )
        effective_start = max(
            transfer.st_time,
            source.ready_time,
            accumulator.ready_time,
        )
        effective_end = effective_start + duration
        source.occurrence.touch(effective_end)
        transfer_src_occurrences[transfer.transfer_id] = source.occurrence
        transfer_input_values[transfer.transfer_id] = source.value
        transfer_effective_times[transfer.transfer_id] = (
            effective_start,
            effective_end,
        )
        combined = accumulator.contributors | contributors
        slot = final_slot(transfer.dst_rank, combined)
        if slot is not None:
            target = _output_token(transfer.dst_rank, slot.offset)
        elif accumulator.occurrence.token[1] == "s":
            target = accumulator.occurrence.token
        else:
            target = new_scratch_token(
                transfer.dst_rank,
                "reduce-{}".format(transfer.transfer_id),
            )
        if accumulator.occurrence.token != target:
            accumulator = copy_state(
                accumulator,
                target,
                effective_start,
                "initialize reduction accumulator",
            )
        accumulator.occurrence.touch(effective_end)
        accumulator.occurrence.close(effective_end)
        if isinstance(accumulator.value, RawValue):
            initial = initial_occurrences.get(accumulator.value.slice_id)
            if initial is not None and target in alias_tokens:
                initial.close(effective_end)
        transfer_accumulator_occurrences[
            transfer.transfer_id
        ] = accumulator.occurrence
        transfer_accumulator_values[transfer.transfer_id] = accumulator.value
        aggregate_versions[logical] += 1
        output_value = AggregateValue(
            logical,
            combined,
            aggregate_versions[logical],
        )
        destination = new_occurrence(
            output_value,
            target,
            effective_end,
        )
        transfer_dst_occurrences[transfer.transfer_id] = destination
        transfer_output_values[transfer.transfer_id] = output_value
        replace_state(
            next(
                state
                for state in states_by_position[
                    (transfer.dst_rank, logical)
                ]
                if state.value == accumulator.value
                and state.contributors == accumulator.contributors
            ),
            _State(
                value=output_value,
                rank=transfer.dst_rank,
                logical=logical,
                contributors=combined,
                occurrence=destination,
                ready_time=effective_end,
                producer_id=transfer.transfer_id,
            ),
        )

    final_values = {}
    final_value_occurrences = {}
    for slot, contributors in sorted(expected_outputs.items()):
        logical_values = {
            logical_slice_index(member, slice_count)
            for member in contributors
        }
        logical = next(iter(logical_values))
        matches = [
            state
            for state in states_by_position[(slot.rank, logical)]
            if state.contributors == contributors
        ]
        if matches:
            state = max(
                matches,
                key=lambda item: (item.ready_time, item.producer_id),
            )
            value = state.value
            target = _output_token(slot.rank, slot.offset)
            if state.occurrence.token == target:
                state.occurrence.final = True
                final_occurrence = state.occurrence
            elif alias_tokens.get(state.occurrence.token) == target:
                state.occurrence.final = True
                final_occurrence = new_occurrence(
                    value,
                    target,
                    state.ready_time,
                    final=True,
                )
            else:
                state = copy_state(
                    state,
                    target,
                    state.ready_time,
                    "place final collective output",
                )
                state.occurrence.final = True
                final_occurrence = state.occurrence
            final_value_occurrences[slot] = final_occurrence
        else:
            aggregate_versions[logical] += int(len(contributors) > 1)
            value = _value_for_contributors(
                contributors,
                logical,
                aggregate_versions[logical],
            )
        final_values[slot] = value

    scratch_offsets, s_chunks = _allocate_scratch(occurrences, rank_count)

    def physical_ref(occurrence: _Occurrence) -> PhysicalRef:
        rank, buffer, token_offset = occurrence.token
        offset = (
            scratch_offsets[occurrence.token]
            if buffer == "s"
            else int(token_offset)
        )
        return PhysicalRef(
            rank=rank,
            buffer=buffer,
            offset=offset,
            valid_from=occurrence.valid_from,
            valid_until=occurrence.valid_until,
        )

    refs_by_occurrence = {
        occurrence.occurrence_id: physical_ref(occurrence)
        for occurrence in occurrences
    }
    value_locations = defaultdict(list)
    for occurrence in occurrences:
        ref = refs_by_occurrence[occurrence.occurrence_id]
        if ref not in value_locations[occurrence.value]:
            value_locations[occurrence.value].append(ref)
    normalized_locations = {
        value: tuple(
            sorted(
                refs,
                key=lambda ref: (
                    ref.rank,
                    ref.buffer,
                    ref.offset,
                    ref.valid_from,
                    ref.valid_until,
                ),
            )
        )
        for value, refs in value_locations.items()
    }

    aliases = tuple(
        (
            PhysicalRef(rank, "i", input_offset, 0.0, math.inf),
            PhysicalRef(rank, "o", output_offset, 0.0, math.inf),
        )
        for rank, input_offset, output_offset in alias_offsets
    )
    final_output_refs = {
        slot: PhysicalRef(slot.rank, "o", slot.offset, 0.0, math.inf)
        for slot in expected_outputs
    }
    local_copies = tuple(
        LocalCopy(
            copy_id=draft.copy_id,
            rank=draft.rank,
            src_ref=refs_by_occurrence[draft.src.occurrence_id],
            dst_ref=refs_by_occurrence[draft.dst.occurrence_id],
            predecessor_state_id=draft.predecessor_state_id,
            st_time=draft.st_time,
            ed_time=draft.ed_time,
            reason=draft.reason,
        )
        for draft in copies
    )
    return BufferPlan(
        value_locations=normalized_locations,
        aliases=aliases,
        local_copies=local_copies,
        i_chunks=i_chunks,
        o_chunks=o_chunks,
        s_chunks=s_chunks,
        slice_count=slice_count,
        initial_refs={
            slice_id: refs_by_occurrence[occurrence.occurrence_id]
            for slice_id, occurrence in initial_occurrences.items()
        },
        final_output_refs=final_output_refs,
        final_values=final_values,
        final_value_refs={
            slot: refs_by_occurrence[occurrence.occurrence_id]
            for slot, occurrence in final_value_occurrences.items()
        },
        transfer_src_refs={
            transfer_id: refs_by_occurrence[occurrence.occurrence_id]
            for transfer_id, occurrence in transfer_src_occurrences.items()
        },
        transfer_dst_refs={
            transfer_id: refs_by_occurrence[occurrence.occurrence_id]
            for transfer_id, occurrence in transfer_dst_occurrences.items()
        },
        transfer_accumulator_refs={
            transfer_id: refs_by_occurrence[occurrence.occurrence_id]
            for transfer_id, occurrence in transfer_accumulator_occurrences.items()
        },
        transfer_input_values=transfer_input_values,
        transfer_accumulator_values=transfer_accumulator_values,
        transfer_output_values=transfer_output_values,
        transfer_effective_times=transfer_effective_times,
    )
