from typing import Dict, Iterable, Tuple

from vericcl.errors import SemanticError
from vericcl.semantics.collective import (
    CollectiveKind,
    CollectiveSpec,
    OutputSlot,
    required_outputs,
)
from vericcl.semantics.state import PayloadState


def _logical_address(
    state: PayloadState,
    rank_count: int,
    slice_count: int,
) -> int:
    global_slice_count = rank_count * slice_count
    for slice_id in state.contributors:
        if slice_id < 0 or slice_id >= global_slice_count:
            raise SemanticError("final state contributor is outside the global range")
    logical_addresses = {
        slice_id % slice_count for slice_id in state.contributors
    }
    if len(logical_addresses) != 1:
        raise SemanticError("final state contributors have different logical addresses")
    logical_address = logical_addresses.pop()
    if state.logical_address != logical_address:
        raise SemanticError("final state has an incorrect logical address")
    return logical_address


def _actual_output(
    spec: CollectiveSpec,
    state: PayloadState,
    rank_count: int,
    slice_count: int,
) -> Tuple[OutputSlot, frozenset]:
    if not state.active:
        raise SemanticError("inactive state cannot be a final output")
    if state.rank < 0 or state.rank >= rank_count:
        raise SemanticError("final state rank is outside the rank range")
    logical_address = _logical_address(state, rank_count, slice_count)
    if spec.kind in {
        CollectiveKind.BROADCAST,
        CollectiveKind.REDUCE,
        CollectiveKind.ALL_REDUCE,
    }:
        slot = OutputSlot(state.rank, logical_address)
    elif spec.kind in {CollectiveKind.GATHER, CollectiveKind.ALL_GATHER}:
        if len(state.contributors) != 1:
            raise SemanticError("gather final state must contain one contributor")
        slot = OutputSlot(state.rank, next(iter(state.contributors)))
    elif spec.kind is CollectiveKind.ALL_TO_ALL:
        if len(state.contributors) != 1:
            raise SemanticError("AllToAll final state must contain one contributor")
        quotient = slice_count // rank_count
        contributor = next(iter(state.contributors))
        source_rank = contributor // slice_count
        offset = source_rank * quotient + logical_address % quotient
        slot = OutputSlot(state.rank, offset)
    elif spec.kind in {CollectiveKind.SCATTER, CollectiveKind.REDUCE_SCATTER}:
        if spec.kind is CollectiveKind.SCATTER and len(state.contributors) != 1:
            raise SemanticError("scatter final state must contain one contributor")
        quotient = slice_count // rank_count
        slot = OutputSlot(state.rank, logical_address % quotient)
    else:
        raise SemanticError("unsupported direct collective")
    return slot, state.contributors


def check_final_states(
    spec: CollectiveSpec,
    rank_count: int,
    slice_count: int,
    states: Iterable[PayloadState],
) -> None:
    expected = required_outputs(spec, rank_count, slice_count)
    actual: Dict[OutputSlot, frozenset] = {}
    state_ids = set()
    for state in states:
        if not isinstance(state, PayloadState):
            raise SemanticError("final states must contain PayloadState values")
        if state.state_id in state_ids:
            raise SemanticError("final state IDs must be unique")
        state_ids.add(state.state_id)
        slot, contributors = _actual_output(
            spec,
            state,
            rank_count,
            slice_count,
        )
        if slot in actual:
            raise SemanticError(
                "duplicate final output at rank {} offset {}".format(
                    slot.rank,
                    slot.offset,
                )
            )
        actual[slot] = contributors

    missing = sorted(set(expected) - set(actual))
    if missing:
        slot = missing[0]
        raise SemanticError(
            "missing final output at rank {} offset {}".format(
                slot.rank,
                slot.offset,
            )
        )
    extra = sorted(set(actual) - set(expected))
    if extra:
        slot = extra[0]
        raise SemanticError(
            "extra final output at rank {} offset {}".format(
                slot.rank,
                slot.offset,
            )
        )
    for slot, contributors in expected.items():
        if actual[slot] != contributors:
            raise SemanticError(
                "final contributors do not match rank {} offset {}".format(
                    slot.rank,
                    slot.offset,
                )
            )
