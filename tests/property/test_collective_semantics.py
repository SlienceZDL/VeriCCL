import pytest
from hypothesis import given, strategies as st

from vericcl.semantics.collective import (
    CollectiveKind,
    CollectiveSpec,
    OutputSlot,
    required_outputs,
)


pytestmark = pytest.mark.phase01


def spec(kind):
    return CollectiveSpec(
        kind=kind,
        datatype="float32",
        reduction_op=(
            "sum"
            if kind
            in {
                CollectiveKind.REDUCE,
                CollectiveKind.ALL_REDUCE,
                CollectiveKind.REDUCE_SCATTER,
            }
            else None
        ),
        root=0 if kind is CollectiveKind.REDUCE else None,
    )


@given(rank_count=st.integers(2, 4), quotient=st.integers(1, 4))
def test_allreduce_outputs_contain_every_source_once(rank_count, quotient):
    slice_count = rank_count * quotient

    outputs = required_outputs(
        spec(CollectiveKind.ALL_REDUCE),
        rank_count,
        slice_count,
    )

    assert len(outputs) == rank_count * slice_count
    for rank in range(rank_count):
        for logical_address in range(slice_count):
            contributors = outputs[OutputSlot(rank, logical_address)]
            assert len(contributors) == rank_count
            assert {item // slice_count for item in contributors} == set(
                range(rank_count)
            )
            assert {item % slice_count for item in contributors} == {
                logical_address
            }


@given(rank_count=st.integers(2, 4), slice_count=st.integers(1, 8))
def test_reduce_outputs_preserve_each_logical_contributor_set(
    rank_count,
    slice_count,
):
    outputs = required_outputs(
        spec(CollectiveKind.REDUCE),
        rank_count,
        slice_count,
    )

    assert set(outputs) == {
        OutputSlot(0, logical_position)
        for logical_position in range(slice_count)
    }
    for logical_position in range(slice_count):
        assert outputs[OutputSlot(0, logical_position)] == frozenset(
            source * slice_count + logical_position
            for source in range(rank_count)
        )


@given(rank_count=st.integers(2, 4), quotient=st.integers(1, 4))
def test_reduce_scatter_partitions_logical_positions(rank_count, quotient):
    slice_count = rank_count * quotient

    outputs = required_outputs(
        spec(CollectiveKind.REDUCE_SCATTER),
        rank_count,
        slice_count,
    )

    assert len(outputs) == slice_count
    recovered_positions = {
        slot.rank * quotient + slot.offset for slot in outputs
    }
    assert recovered_positions == set(range(slice_count))
    for slot, contributors in outputs.items():
        logical_address = slot.rank * quotient + slot.offset
        assert contributors == frozenset(
            rank * slice_count + logical_address
            for rank in range(rank_count)
        )
