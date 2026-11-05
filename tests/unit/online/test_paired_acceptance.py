from fractions import Fraction
from math import comb

import pytest

from vericcl.errors import SemanticError
from vericcl.verification.online.acceptance import (
    AcceptanceContext, LaunchSample, compare_measured, one_sided_binomial,
)
from vericcl.verification.online.statistical_ledger import StatisticalLedger


def ledger(tmp_path, **kwargs):
    return StatisticalLedger(tmp_path / 'attempts.jsonl', task_id='task', **kwargs)


def sample(value, request):
    return LaunchSample(value, True, True, False, request.launch_id, 'unit_fixture')


def test_default_off_does_not_allocate_or_launch(tmp_path):
    book = ledger(tmp_path)
    decision = compare_measured('old', 'new', book, AcceptanceContext(enabled=False))
    assert decision.status == 'disabled' and not decision.accepted and book.attempts() == ()


def test_twenty_balanced_adjacent_pairs_and_trace_disabled(tmp_path):
    calls = []
    def driver(candidate, request):
        calls.append((candidate, request))
        assert request.warmup == 5 and request.timed == 20 and not request.trace_enabled
        return sample(100 if candidate == 'old' else 90, request)
    book = ledger(tmp_path)
    decision = compare_measured('old', 'new', book, AcceptanceContext(True, driver, seed=7, allow_test_samples=True))
    assert decision.accepted and decision.pair_count == 20 and decision.selected_round == 1
    assert len(calls) == 40
    pairs = [tuple(calls[i+j][0] for j in (0, 1)) for i in range(0, 40, 2)]
    assert pairs.count(('old', 'new')) == pairs.count(('new', 'old')) == 10
    assert decision.p_value == float(Fraction(1, 2**20))
    assert decision.alpha == .025 and decision.median_gain == pytest.approx(.1)
    events = book.events()
    assert events[1]['event'] == 'attempt_reserved'
    assert events[-1]['event'] == 'attempt_finished'
    assert sum(event['event'] == 'sample' for event in events) == 40


def test_binomial_uses_all_pairs_ties_nonwins_and_independent_exact_sum():
    for n in (2, 20, 24):
        for wins in range(n + 1):
            expected = sum(Fraction(comb(n, k), 2**n) for k in range(wins, n+1))
            assert one_sided_binomial(wins, n) == float(expected)
    with pytest.raises(SemanticError):
        one_sided_binomial(21, 20)


def test_threshold_equality_is_not_a_win(tmp_path):
    def driver(candidate, request):
        return sample(100 if candidate == 'old' else 100, request)
    result = compare_measured('old', 'new', ledger(tmp_path), AcceptanceContext(True, driver, seed=1, allow_test_samples=True))
    assert result.status == 'no_measured_gain' and result.wins == 0 and result.p_value == 1


def test_both_groups_must_be_stable_in_same_round_and_first_is_used(tmp_path):
    def driver(candidate, request):
        baseline = 100 if candidate == 'old' else 80
        if (request.round_index == 1 and candidate == 'old') or (request.round_index == 2 and candidate == 'new'):
            baseline += 30 * (request.pair_index % 2)
        return sample(baseline, request)
    book = ledger(tmp_path)
    result = compare_measured('old', 'new', book, AcceptanceContext(True, driver, seed=2, allow_test_samples=True))
    assert result.accepted and result.selected_round == 3
    assert len([x for x in book.events() if x['event'] == 'sample']) == 120


def test_unstable_and_failed_attempts_consume_index_across_resume(tmp_path):
    def unstable(candidate, request):
        return sample(100 + request.pair_index % 2 * 100, request)
    book = ledger(tmp_path)
    result = compare_measured('old', 'new', book, AcceptanceContext(True, unstable, seed=2, allow_test_samples=True))
    assert result.status == 'unstable' and result.attempt_index == 1
    def failing(candidate, request):
        raise RuntimeError('launch failed')
    resumed = ledger(tmp_path)
    failed = compare_measured('old', 'new2', resumed, AcceptanceContext(True, failing, seed=3))
    assert failed.status == 'measurement_failed' and failed.attempt_index == 2
    plan = ledger(tmp_path).reserve('old', 'new3', seed=4)
    assert plan.attempt_index == 3 and plan.alpha == pytest.approx(.05 / 12)
    assert sum(p.alpha for p in book.attempts()) <= .05


def test_sample_count_grows_only_before_data_and_plan_is_immutable(tmp_path):
    book = ledger(tmp_path, single_attempt_cap=2**-25)
    plan = book.reserve('a', 'b', seed=10)
    assert plan.pair_count == 26  # even and the all-win p-value can reach alpha
    assert len(plan.orders) == 3 and all(len(order) == 26 for order in plan.orders)
    assert all(order.count('AB') == order.count('BA') == 13 for order in plan.orders)
    with pytest.raises(SemanticError):
        StatisticalLedger(book.path, task_id='task', single_attempt_cap=.05)
    with pytest.raises(SemanticError):
        StatisticalLedger(book.path, task_id='different', single_attempt_cap=2**-25)


def test_invalid_or_traced_sample_cannot_pass_enabled_gate(tmp_path):
    def traced(candidate, request):
        return LaunchSample(1, True, True, True, request.launch_id, 'unit_fixture')
    result = compare_measured('a', 'b', ledger(tmp_path), AcceptanceContext(True, traced, seed=0))
    assert not result.accepted and result.status == 'measurement_failed'


def test_incomplete_attempt_is_not_reused_or_refunded(tmp_path):
    first = ledger(tmp_path).reserve('a', 'b', seed=3)
    second = ledger(tmp_path).reserve('a', 'c', seed=3)
    assert first.attempt_index == 1 and second.attempt_index == 2
    assert first.orders != second.orders


def test_truncated_ledger_fails_closed_without_reset(tmp_path):
    book = ledger(tmp_path)
    book.reserve('a', 'b', seed=3)
    with book.path.open('ab') as output:
        output.write(b'{"interrupted":')
    from vericcl.execution.context import StorageFailure
    with pytest.raises(StorageFailure):
        ledger(tmp_path)


def test_nonfinite_sample_is_retained_as_failure_not_storage_failure(tmp_path):
    def invalid(candidate, request):
        return sample(float('nan'), request)
    book = ledger(tmp_path)
    result = compare_measured('a', 'b', book, AcceptanceContext(True, invalid, seed=1, allow_test_samples=True))
    assert result.status == 'measurement_failed'
    assert book.events()[2]['payload']['sample']['time_us'] == 'nan'
    assert len(book.attempts()) == 1


def test_nonzero_threshold_equality_and_configured_protocol_are_preplanned(tmp_path):
    def driver(candidate, request):
        assert request.warmup == 7 and request.timed == 25
        return sample(100 if candidate == 'a' else 90, request)
    book = ledger(tmp_path)
    result = compare_measured('a', 'b', book, AcceptanceContext(True, driver, seed=1,
        minimum_gain=.1, warmup=7, timed=25, max_rounds=2, allow_test_samples=True))
    assert not result.accepted and result.wins == 0 and result.p_value == 1
    assert len(book.attempts()[0].orders) == 2


def test_launch_identity_and_runtime_change_fail_closed(tmp_path):
    def driver(candidate, request):
        return LaunchSample(100, True, True, False, request.launch_id, 'native_msccl',
                            runtime_identity='different-on-B' if request.side == 'B' else 'runtime')
    result = compare_measured('a', 'b', ledger(tmp_path), AcceptanceContext(True, driver, seed=1))
    assert result.status == 'measurement_failed' and 'identity changed' in result.issues[0]


def test_sampling_configuration_rejects_unbounded_retry_and_relaxed_alpha(tmp_path):
    with pytest.raises(SemanticError):
        ledger(tmp_path, task_alpha=.1)
    book = ledger(tmp_path)
    with pytest.raises(SemanticError):
        book.reserve('a', 'b', seed=1, max_rounds=4)
    with pytest.raises(SemanticError):
        book.reserve('a', 'b', seed=1, pair_count=21)
    assert not book.attempts()
