import pytest
from hypothesis import given, strategies as st

from vericcl.errors import SemanticError


@given(st.integers(min_value=2, max_value=8), st.permutations(tuple(range(8))))
def test_reduction_tree_order_preserves_exactly_one_contribution(ranks, order):
    from vericcl.semantics.ranges import DataRange
    from vericcl.semantics.versions import BufferRegion, StateVersion, VersionLedger
    ledger = VersionLedger(scratch_budgets={0: 4096})
    for source in range(ranks):
        ledger.add_initial(StateVersion("s" + str(source), 0, 0, 4,
                           (DataRange(source, 0, 4),), BufferRegion(0, "s" + str(source), 0, 4), 0))
    selected = [r for r in order if r < ranks]
    current = "s" + str(selected[0])
    expected = {selected[0]}
    for step, source in enumerate(selected[1:], 1):
        expected.add(source)
        result = ledger.reduce(current, "s" + str(source), state_id="a" + str(step),
                               destination=BufferRegion(0, "result", 0, 4), start_us=step, end_us=step + 1)
        current = result.state_id
        # Independent byte-level reference, not the ledger's merge utility.
        actual = [(piece.source_rank, offset) for piece in result.contributors for offset in range(piece.start, piece.end)]
        assert sorted(actual) == sorted((rank, byte) for rank in expected for byte in range(4))


def test_duplicate_contribution_and_different_logical_positions_are_rejected():
    from vericcl.semantics.ranges import DataRange
    from vericcl.semantics.versions import BufferRegion, StateVersion, VersionLedger
    ledger = VersionLedger(scratch_budgets={0: 1024})
    p = StateVersion("p", 0, 0, 4, (DataRange(0, 0, 4),), BufferRegion(0, "p", 0, 4), 0)
    ledger.add_initial(p)
    ledger.copy("p", state_id="duplicate", destination=BufferRegion(0, "q", 0, 4), start_us=0, end_us=1)
    with pytest.raises(SemanticError, match="contribution"):
        ledger.reduce("p", "duplicate", state_id="bad", destination=p.storage, start_us=1, end_us=2)
    other = StateVersion("other", 0, 4, 8, (DataRange(1, 4, 8),), BufferRegion(0, "o", 0, 4), 0)
    ledger.add_initial(other)
    with pytest.raises(SemanticError, match="logical"):
        ledger.reduce("p", "other", state_id="bad", destination=p.storage, start_us=1, end_us=2)


def test_copy_keeps_the_source_available_for_two_consumers():
    from vericcl.semantics.ranges import DataRange
    from vericcl.semantics.versions import BufferRegion, StateVersion, VersionLedger
    ledger = VersionLedger(scratch_budgets={0: 0, 1: 0, 2: 0})
    ledger.add_initial(StateVersion("p", 0, 0, 4, (DataRange(0, 0, 4),), BufferRegion(0, "input", 0, 4), 0))
    a = ledger.copy("p", state_id="a", destination=BufferRegion(1, "output", 0, 4), start_us=0, end_us=1)
    b = ledger.copy("p", state_id="b", destination=BufferRegion(2, "output", 0, 4), start_us=0, end_us=1)
    assert ledger.is_active("p")
    assert a.contributors == b.contributors == (DataRange(0, 0, 4),)
