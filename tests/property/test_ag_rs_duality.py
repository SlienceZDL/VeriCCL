import pytest
from hypothesis import given, strategies as st

from vericcl.composer.dual import reverse_allgather_schedule

from tests.unit.composer.helpers import (
    reduce_spec,
    reduce_target,
    virtual_reduce_chain,
    virtual_reduce_star,
)


pytestmark = pytest.mark.phase03


@given(st.integers(min_value=2, max_value=4))
def test_chain_duality_preserves_every_contributor_once(rank_count):
    reduced = reverse_allgather_schedule(
        virtual_reduce_chain(rank_count),
        reduce_spec(),
        reduce_target(rank_count),
    )
    root_transfers = [
        transfer for transfer in reduced.transfers if transfer.dst_rank == 0
    ]

    assert len(root_transfers) == 1
    assert root_transfers[0].member_slice_ids == frozenset(
        range(1, rank_count)
    )
    assert {
        atom.slice_id for atom in root_transfers[0].atoms
    } == root_transfers[0].member_slice_ids
    assert all(
        transfer.kind == "REDUCE" for transfer in reduced.transfers
    )


@given(st.integers(min_value=2, max_value=4))
def test_star_duality_never_reuses_one_accumulator_version(rank_count):
    reduced = reverse_allgather_schedule(
        virtual_reduce_star(rank_count),
        reduce_spec(),
        reduce_target(rank_count),
    )
    ordered = sorted(
        reduced.transfers,
        key=lambda transfer: (transfer.st_time, transfer.transfer_id),
    )
    accumulated = frozenset({0})
    previous = None
    for transfer in ordered:
        assert not accumulated & transfer.member_slice_ids
        accumulated |= transfer.member_slice_ids
        assert frozenset(
            reduced.metadata["tree_contributors"][transfer.transfer_id]
        ) == accumulated
        if previous is not None:
            assert previous.transfer_id in transfer.predecessor_ids
        previous = transfer
    assert accumulated == frozenset(range(rank_count))
