"""Storage lifetime and work bounds for large, disjoint state partitions."""
import pytest

from vericcl.errors import SemanticError
from vericcl.semantics.ranges import DataRange
from vericcl.semantics.versions import BufferRegion, StateVersion, VersionLedger


def initial(name, offset, size=4, *, buffer='arena', kind='scratch', rank=0):
    return StateVersion(name, rank, 0, size, (DataRange(rank, 0, size),),
                        BufferRegion(rank, buffer, offset, size, kind), 0)


def test_disjoint_initial_partition_does_not_compare_every_existing_state(monkeypatch):
    calls = 0
    original = BufferRegion.overlaps

    def counted(self, other):
        nonlocal calls
        calls += 1
        return original(self, other)

    monkeypatch.setattr(BufferRegion, 'overlaps', counted)
    ledger = VersionLedger(scratch_budgets={0: 4096})
    count = 256
    for index in range(count):
        ledger.add_initial(initial('s' + str(index), 4 * index))
    assert all(ledger.is_active('s' + str(index)) for index in range(count))
    assert calls <= 4 * count, 'disjoint storage insertion must not perform all-pairs overlap scans'


def test_partial_overwrite_invalidates_whole_states_and_preserves_adjacent_regions():
    ledger = VersionLedger(scratch_budgets={0: 16})
    ledger.add_initial(initial('source', 0, 8, buffer='input', kind='input'))
    for index in (2, 0, 3, 1):
        ledger.add_initial(initial('s' + str(index), 4 * index))
    ledger.copy('source', state_id='wide', destination=BufferRegion(0, 'arena', 2, 8, 'scratch'),
                start_us=0, end_us=1)
    assert not any(ledger.is_active('s' + str(index)) for index in (0, 1, 2))
    assert ledger.is_active('s3') and ledger.is_active('wide')
    # Three whole versions were invalidated: 16 - 12 + 8 = 12 live scratch bytes.
    ledger.add_initial(initial('adjacent', 16))
    with pytest.raises(SemanticError, match='scratch'):
        ledger.add_initial(initial('over_budget', 20))
    ledger.release('wide', at_us=1)
    ledger.add_initial(initial('reused', 0, 8))


def test_failed_write_leaves_storage_budget_identity_and_reader_state_unchanged():
    ledger = VersionLedger(scratch_budgets={0: 8})
    ledger.add_initial(initial('source', 0, buffer='input', kind='input'))
    ledger.add_initial(initial('a', 0))
    ledger.add_initial(initial('b', 4))
    with pytest.raises(SemanticError, match='scratch'):
        ledger.copy('source', state_id='retry', destination=BufferRegion(0, 'arena', 8, 4, 'scratch'),
                    start_us=0, end_us=5)
    ledger.release('a', at_us=0)
    ledger.copy('source', state_id='retry', destination=BufferRegion(0, 'arena', 8, 4, 'scratch'),
                start_us=0, end_us=1)
    ledger.release('source', at_us=1)
    assert ledger.is_active('b') and ledger.is_active('retry')
    assert not ledger.is_active('source')


def test_reader_maximum_remains_exact_across_different_buffers_and_release():
    ledger = VersionLedger(scratch_budgets={0: 8, 1: 8})
    for rank in (0, 1):
        ledger.add_initial(initial('s' + str(rank), 0, rank=rank))
    ledger.reserve_read('s0', 'late', start_us=0, end_us=7)
    ledger.reserve_read('s0', 'early', start_us=0, end_us=2)
    ledger.reserve_read('s1', 'other', start_us=0, end_us=20)
    with pytest.raises(SemanticError, match='reader'):
        ledger.release('s0', at_us=6)
    ledger.release('s0', at_us=7)
    ledger.add_initial(initial('new', 0))
    assert ledger.is_active('s1') and ledger.is_active('new')


@pytest.mark.parametrize('offset', [0, 2, 6])
def test_initial_overlap_rejection_does_not_poison_a_later_nonoverlapping_insert(offset):
    ledger = VersionLedger(scratch_budgets={0: 32})
    ledger.add_initial(initial('original', 2, 8))
    with pytest.raises(SemanticError, match='overlap'):
        ledger.add_initial(initial('retry', offset))
    ledger.add_initial(initial('retry', 10))
    assert ledger.is_active('original') and ledger.is_active('retry')
