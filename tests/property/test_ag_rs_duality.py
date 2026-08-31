from collections import Counter
from dataclasses import replace

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
    virtual = virtual_reduce_star(rank_count)
    metadata = dict(virtual.metadata)
    metadata["routing_only"] = True
    reduced = reverse_allgather_schedule(
        replace(virtual, metadata=metadata),
        reduce_spec(),
        reduce_target(rank_count),
    )
    counts = Counter(
        member
        for transfer in reduced.transfers
        for member in transfer.member_slice_ids
    )
    key = "r00000000-o00000000"
    ordered = tuple(
        sorted(
            reduced.transfers,
            key=lambda transfer: (transfer.st_time, transfer.transfer_id),
        )
    )
    transfer_ids = {transfer.transfer_id for transfer in ordered}
    consumed = [
        state_id
        for transition in reduced.metadata["aggregate_consumptions"].values()
        for state_id in transition["consumed_state_ids"]
    ]

    assert counts == Counter({rank: 1 for rank in range(1, rank_count)})
    assert reduced.metadata["final_dependencies"][key] == (
        ordered[-1].transfer_id,
    )
    assert set(reduced.metadata["aggregate_consumptions"]) == transfer_ids
    assert len(consumed) == len(set(consumed))
