from collections import Counter

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


@given(st.integers(min_value=2, max_value=5))
def test_star_duality_consumes_each_remote_contributor_once(rank_count):
    reduced = reverse_allgather_schedule(
        virtual_reduce_star(rank_count),
        reduce_spec(),
        reduce_target(rank_count),
    )
    counts = Counter(
        member
        for transfer in reduced.transfers
        for member in transfer.member_slice_ids
    )
    key = "r00000000-o00000000"
    dependencies = tuple(
        sorted(transfer.transfer_id for transfer in reduced.transfers)
    )

    assert counts == Counter({rank: 1 for rank in range(1, rank_count)})
    assert reduced.metadata["final_dependencies"][key] == dependencies
    assert reduced.metadata["aggregate_consumptions"][key] == dependencies
