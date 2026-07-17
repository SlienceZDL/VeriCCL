import pytest

from vericcl.semantics.collective import CollectiveKind
from vericcl.xml.buffers import build_buffer_plan

from tests.unit.xml.helpers import (
    final_schedule,
    inplace_alltoall_overwrite_schedule,
    reduce_chain_schedule,
    resolved,
)


pytestmark = pytest.mark.phase04


def _alias_offsets(plan, rank):
    return {
        (left.buffer, left.offset, right.buffer, right.offset)
        for left, right in plan.aliases
        if left.rank == rank
    }


def test_operator_specific_inplace_aliases():
    allreduce = build_buffer_plan(
        final_schedule(CollectiveKind.ALL_REDUCE, inplace=True),
        resolved(CollectiveKind.ALL_REDUCE, inplace=True),
    )
    allgather = build_buffer_plan(
        final_schedule(CollectiveKind.ALL_GATHER, inplace=True),
        resolved(CollectiveKind.ALL_GATHER, inplace=True),
    )
    reduce = build_buffer_plan(
        final_schedule(CollectiveKind.REDUCE, inplace=True),
        resolved(CollectiveKind.REDUCE, inplace=True),
    )
    reduce_scatter = build_buffer_plan(
        final_schedule(CollectiveKind.REDUCE_SCATTER, inplace=True),
        resolved(CollectiveKind.REDUCE_SCATTER, inplace=True),
    )

    assert ("i", 1, "o", 1) in _alias_offsets(allreduce, 1)
    assert ("i", 1, "o", 3) in _alias_offsets(allgather, 1)
    assert ("i", 1, "o", 1) in _alias_offsets(reduce, 0)
    assert _alias_offsets(reduce, 1) == set()
    assert ("i", 1, "o", 0) in _alias_offsets(reduce_scatter, 1)


def test_inplace_alltoall_preserves_input_before_receive_overwrite():
    schedule = inplace_alltoall_overwrite_schedule()
    plan = build_buffer_plan(
        schedule,
        resolved(CollectiveKind.ALL_TO_ALL, inplace=True),
    )

    preservation = [
        copy
        for copy in plan.local_copies
        if copy.reason == "preserve live in-place input"
    ]
    assert len(preservation) == 2
    rank_one_copy = next(copy for copy in preservation if copy.rank == 1)
    assert rank_one_copy.src_ref.buffer_offset == ("i", 0)
    assert rank_one_copy.dst_ref.buffer == "s"
    assert plan.transfer_src_refs["outgoing"] == rank_one_copy.dst_ref


@pytest.mark.parametrize("overlap,expected_offsets", [(False, {0}), (True, {0, 1})])
def test_scratch_first_fit_reuses_only_non_overlapping_intervals(
    overlap,
    expected_offsets,
):
    schedule = reduce_chain_schedule(slices=2, overlap=overlap)
    plan = build_buffer_plan(
        schedule,
        resolved(CollectiveKind.REDUCE, ranks=3, slices=2),
    )

    actual = {
        ref.offset
        for refs in plan.value_locations.values()
        for ref in refs
        if ref.rank == 1 and ref.buffer == "s"
    }
    assert actual == expected_offsets
