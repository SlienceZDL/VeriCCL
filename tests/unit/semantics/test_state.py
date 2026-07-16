import pytest

from vericcl.errors import SemanticError
from vericcl.semantics.state import (
    PayloadLedger,
    PayloadState,
    initial_payload_states,
)


pytestmark = pytest.mark.phase01


def state(
    state_id,
    rank,
    logical_address,
    contributors,
    *,
    version=0,
    ready_time=0.0,
    active=True,
):
    contributor_set = frozenset(contributors)
    return PayloadState(
        state_id=state_id,
        version=version,
        rank=rank,
        logical_address=logical_address,
        contributors=contributor_set,
        ready_time=ready_time,
        active=active,
        member_paths=tuple(
            (slice_id, ()) for slice_id in sorted(contributor_set)
        ),
    )


def ledger_with_states(*states):
    return PayloadLedger(states)


def ledger_with_three_singletons():
    return ledger_with_states(
        state("a", 0, 0, {0}),
        state("b", 0, 0, {4}),
        state("c", 0, 0, {8}),
    )


def test_initial_payload_states_use_global_slice_identity():
    states = initial_payload_states(rank_count=2, slice_count=4)

    assert len(states) == 8
    assert states[0].rank == 0
    assert states[0].logical_address == 0
    assert states[0].contributors == frozenset({0})
    assert states[7].rank == 1
    assert states[7].logical_address == 3
    assert states[7].contributors == frozenset({7})


def test_reduce_unions_disjoint_contributors():
    ledger = ledger_with_states(
        state("a", 0, 0, {0}),
        state("b", 0, 0, {4}),
    )

    result = ledger.reduce("a", "b", dst_rank=0, ready_time=3.0)

    assert result.contributors == frozenset({0, 4})
    assert result.rank == 0
    assert result.ready_time == 3.0
    assert ledger.state("a").active is False
    assert ledger.state("b").active is False


def test_reduce_rejects_intersecting_contributors():
    ledger = ledger_with_states(
        state("a", 0, 0, {0, 4}),
        state("b", 1, 0, {4}),
    )

    with pytest.raises(SemanticError, match="contributors must be disjoint"):
        ledger.reduce("a", "b", dst_rank=1, ready_time=3.0)


def test_reduce_requires_equal_logical_addresses():
    ledger = ledger_with_states(
        state("a", 0, 0, {0}),
        state("b", 0, 1, {5}),
    )

    with pytest.raises(SemanticError, match="logical address"):
        ledger.reduce("a", "b", dst_rank=0, ready_time=3.0)


def test_reduce_ready_time_must_follow_both_inputs():
    ledger = ledger_with_states(
        state("a", 0, 0, {0}, ready_time=1.0),
        state("b", 0, 0, {4}, ready_time=4.0),
    )

    with pytest.raises(SemanticError, match="ready_time"):
        ledger.reduce("a", "b", dst_rank=0, ready_time=3.0)


def test_consumed_reduce_source_cannot_be_reused():
    ledger = ledger_with_three_singletons()
    ledger.reduce("a", "b", dst_rank=0, ready_time=2.0)

    with pytest.raises(SemanticError, match="state version is inactive"):
        ledger.reduce("a", "c", dst_rank=0, ready_time=4.0)


def test_active_aggregate_is_unique_per_rank_and_logical_address():
    ledger = ledger_with_states(
        state("aggregate", 1, 0, {0, 4}),
        state("a", 1, 0, {8}),
        state("b", 0, 0, {12}),
    )

    with pytest.raises(SemanticError, match="active aggregate already exists"):
        ledger.reduce("a", "b", dst_rank=1, ready_time=2.0)


def test_incomplete_state_has_at_most_one_outbound_send():
    ledger = ledger_with_states(state("a", 0, 0, {0}))
    required = frozenset({0, 4})
    result = ledger.send("a", dst_rank=1, ready_time=1.0, required_contributors=required)

    assert result.contributors == frozenset({0})
    assert ledger.state("a").active is False
    with pytest.raises(SemanticError, match="incomplete state already sent"):
        ledger.send(
            "a",
            dst_rank=2,
            ready_time=1.0,
            required_contributors=required,
        )


def test_complete_state_can_branch_without_consuming_source():
    ledger = ledger_with_states(state("a", 0, 0, {0, 4}))
    required = frozenset({0, 4})

    left = ledger.send(
        "a",
        dst_rank=1,
        ready_time=1.0,
        required_contributors=required,
    )
    right = ledger.send(
        "a",
        dst_rank=2,
        ready_time=1.5,
        required_contributors=required,
    )

    assert ledger.state("a").active is True
    assert left.state_id != right.state_id
    assert left.member_paths == right.member_paths


def test_singleton_can_coexist_with_active_aggregate_at_destination():
    ledger = ledger_with_states(
        state("aggregate", 1, 0, {0, 4}),
        state("singleton", 0, 0, {8}),
    )

    result = ledger.send(
        "singleton",
        dst_rank=1,
        ready_time=1.0,
        required_contributors=frozenset({8}),
    )

    assert result.rank == 1
    assert result.contributors == frozenset({8})


def test_send_rejects_contributors_outside_required_set():
    ledger = ledger_with_states(state("a", 0, 0, {0, 4}))

    with pytest.raises(SemanticError, match="required contributors"):
        ledger.send(
            "a",
            dst_rank=1,
            ready_time=1.0,
            required_contributors=frozenset({0}),
        )


def test_merge_local_records_no_network_transfer():
    ledger = ledger_with_states(
        state("incoming", 0, 0, {4}),
        state("local", 0, 0, {0}),
    )
    transfers = []

    result = ledger.merge_local("incoming", "local", ready_time=2.0)

    assert result.contributors == frozenset({0, 4})
    assert transfers == []


def test_merge_local_requires_both_states_at_same_rank():
    ledger = ledger_with_states(
        state("incoming", 0, 0, {4}),
        state("local", 1, 0, {0}),
    )

    with pytest.raises(SemanticError, match="same rank"):
        ledger.merge_local("incoming", "local", ready_time=2.0)
