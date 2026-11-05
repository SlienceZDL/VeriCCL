from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Tuple

from vericcl.errors import SemanticError
from vericcl.input.models import ResolvedInput
from vericcl.semantics.atom import Schedule
from vericcl.semantics.collective import required_outputs
from vericcl.xml.model import (
    AggregateValue,
    BufferPlan,
    PhysicalRef,
    RawValue,
    ValueKey,
)


Address = Tuple[int, str, int]


class _Aliases:
    def __init__(self, plan: BufferPlan) -> None:
        self._parent: Dict[Address, Address] = {}
        for left, right in plan.aliases:
            self.union(self._address(left), self._address(right))

    @staticmethod
    def _address(ref: PhysicalRef) -> Address:
        return ref.rank, ref.buffer, ref.offset

    def find(self, address: Address) -> Address:
        self._parent.setdefault(address, address)
        parent = self._parent[address]
        if parent != address:
            self._parent[address] = self.find(parent)
        return self._parent[address]

    def union(self, left: Address, right: Address) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            root, child = sorted((left_root, right_root))
            self._parent[child] = root

    def canonical(self, ref: PhysicalRef) -> Address:
        return self.find(self._address(ref))


def _contributors(value: ValueKey) -> frozenset[int]:
    if isinstance(value, RawValue):
        return frozenset({value.slice_id})
    return value.contributors


def _all_refs(plan: BufferPlan) -> Iterable[PhysicalRef]:
    for refs in plan.value_locations.values():
        yield from refs
    for pair in plan.aliases:
        yield from pair
    for copy in plan.local_copies:
        yield copy.src_ref
        yield copy.dst_ref
    yield from plan.initial_refs.values()
    yield from plan.final_output_refs.values()
    yield from plan.final_value_refs.values()
    yield from plan.transfer_src_refs.values()
    yield from plan.transfer_dst_refs.values()
    yield from plan.transfer_accumulator_refs.values()


def _verify_bounds(plan: BufferPlan, ref: PhysicalRef) -> None:
    counts = {
        "i": plan.i_chunks,
        "o": plan.o_chunks,
        "s": plan.s_chunks,
    }[ref.buffer]
    if ref.rank not in counts or ref.offset >= counts[ref.rank]:
        raise SemanticError("buffer offset is outside declared chunks")


def _overlap(left: PhysicalRef, right: PhysicalRef) -> bool:
    return (
        left.valid_from < right.valid_until
        and right.valid_from < left.valid_until
    )


def _covers(ref: PhysicalRef, start: float, end: float) -> bool:
    return ref.valid_from <= start and ref.valid_until >= end


def verify_buffer_liveness(
    schedule: Schedule,
    plan: BufferPlan,
    inputs: ResolvedInput,
) -> None:
    if not isinstance(schedule, Schedule):
        raise SemanticError("schedule must be a Schedule")
    if not isinstance(plan, BufferPlan):
        raise SemanticError("plan must be a BufferPlan")
    if not isinstance(inputs, ResolvedInput):
        raise SemanticError("inputs must be a ResolvedInput")
    if schedule.rank_count != inputs.rank_count:
        raise SemanticError("schedule and input rank counts differ")
    if schedule.slice_count != plan.slice_count:
        raise SemanticError("schedule and buffer plan slice counts differ")

    for ref in _all_refs(plan):
        _verify_bounds(plan, ref)

    aliases = _Aliases(plan)
    locations = defaultdict(list)
    for value, refs in plan.value_locations.items():
        for ref in refs:
            locations[aliases.canonical(ref)].append((value, ref))
    for entries in locations.values():
        for index, (left_value, left_ref) in enumerate(entries):
            for right_value, right_ref in entries[index + 1 :]:
                if left_value != right_value and _overlap(left_ref, right_ref):
                    raise SemanticError(
                        "live values share a physical location"
                    )

    transfers = {transfer.transfer_id: transfer for transfer in schedule.transfers}
    if set(plan.transfer_src_refs) != set(transfers):
        raise SemanticError("every transfer requires one source reference")
    if set(plan.transfer_dst_refs) != set(transfers):
        raise SemanticError("every transfer requires one destination reference")
    if set(plan.transfer_input_values) != set(transfers):
        raise SemanticError("every transfer requires one input value")
    if set(plan.transfer_output_values) != set(transfers):
        raise SemanticError("every transfer requires one output value")
    if set(plan.transfer_effective_times) != set(transfers):
        raise SemanticError("every transfer requires effective timing")

    for transfer_id, transfer in transfers.items():
        effective_start, effective_end = plan.transfer_effective_times[
            transfer_id
        ]
        if (
            effective_start < transfer.st_time
            or effective_end < effective_start
            or effective_end - effective_start
            != transfer.ed_time - transfer.st_time
        ):
            raise SemanticError("transfer effective timing is invalid")
        src_ref = plan.transfer_src_refs[transfer_id]
        dst_ref = plan.transfer_dst_refs[transfer_id]
        if src_ref.rank != transfer.src_rank or dst_ref.rank != transfer.dst_rank:
            raise SemanticError("transfer reference rank is incorrect")
        if not _covers(src_ref, effective_start, effective_end):
            raise SemanticError("transfer reads a value outside its live interval")
        if dst_ref.valid_from > effective_end:
            raise SemanticError("transfer destination becomes valid too late")
        input_value = plan.transfer_input_values[transfer_id]
        if _contributors(input_value) != transfer.member_slice_ids:
            raise SemanticError("transfer input contributors are incorrect")
        output_value = plan.transfer_output_values[transfer_id]
        if transfer.kind == "SEND":
            if output_value != input_value:
                raise SemanticError("SEND must preserve its value identity")
            if transfer_id in plan.transfer_accumulator_refs:
                raise SemanticError("SEND must not define an accumulator")
            if transfer_id in plan.transfer_accumulator_values:
                raise SemanticError("SEND must not define an accumulator value")
        else:
            accumulator = plan.transfer_accumulator_refs.get(transfer_id)
            if accumulator is None:
                raise SemanticError("REDUCE requires an initialized accumulator")
            if aliases.canonical(accumulator) != aliases.canonical(dst_ref):
                raise SemanticError(
                    "REDUCE accumulator and destination addresses differ"
                )
            if not _covers(accumulator, effective_start, effective_end):
                raise SemanticError(
                    "rrc destination accumulator is not live before reduction"
                )
            accumulator_value = plan.transfer_accumulator_values.get(transfer_id)
            if accumulator_value is None:
                raise SemanticError("REDUCE requires an accumulator value")
            if not isinstance(output_value, AggregateValue):
                raise SemanticError("REDUCE must produce an aggregate value")
            if (
                not _contributors(accumulator_value).isdisjoint(
                    transfer.member_slice_ids
                )
                or _contributors(accumulator_value) | transfer.member_slice_ids
                != output_value.contributors
            ):
                raise SemanticError(
                    "REDUCE output must add disjoint accumulator contributors"
                )

    for copy in plan.local_copies:
        if not _covers(copy.src_ref, copy.st_time, copy.ed_time):
            raise SemanticError("local copy source is not live")
        if copy.dst_ref.valid_from > copy.ed_time:
            raise SemanticError("local copy destination becomes valid too late")

    if not inputs.collective.inplace:
        if any(ref.buffer == "i" for ref in plan.transfer_dst_refs.values()):
            raise SemanticError("out-of-place input buffer is modified")
        if any(copy.dst_ref.buffer == "i" for copy in plan.local_copies):
            raise SemanticError("out-of-place input buffer is modified")

    for transfer_id, transfer in transfers.items():
        effective_start, effective_end = plan.transfer_effective_times[
            transfer_id
        ]
        dst_ref = plan.transfer_dst_refs[transfer_id]
        canonical = aliases.canonical(dst_ref)
        accumulator = plan.transfer_accumulator_refs.get(transfer_id)
        for value, live_ref in locations[canonical]:
            if accumulator is not None and live_ref == accumulator:
                continue
            if (
                accumulator is not None
                and _contributors(value).isdisjoint(
                    transfer.member_slice_ids
                )
                and _contributors(value) | transfer.member_slice_ids
                == _contributors(plan.transfer_output_values[transfer_id])
            ):
                continue
            if value == plan.transfer_output_values[transfer_id]:
                continue
            if (
                live_ref.valid_from < effective_end
                and effective_start < live_ref.valid_until
            ):
                raise SemanticError("transfer write overlaps a live value")

    expected = required_outputs(
        inputs.collective,
        schedule.rank_count,
        schedule.slice_count,
    )
    if set(plan.final_output_refs) != set(expected):
        raise SemanticError("final output references do not match the collective")
    if set(plan.final_values) != set(expected):
        raise SemanticError("final values do not match the collective")
    if set(plan.final_value_refs) != set(expected):
        raise SemanticError("a final output value is uninitialized")
    for slot, contributors in expected.items():
        contract_ref = plan.final_output_refs[slot]
        actual_ref = plan.final_value_refs[slot]
        if contract_ref.buffer_offset != ("o", slot.offset):
            raise SemanticError("final output contract address is incorrect")
        if actual_ref.rank != slot.rank or actual_ref.buffer_offset != (
            "o",
            slot.offset,
        ):
            raise SemanticError("final value is stored at an incorrect address")
        if _contributors(plan.final_values[slot]) != contributors:
            raise SemanticError("final output contributors are incorrect")
