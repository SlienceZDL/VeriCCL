import math
from dataclasses import replace

import pytest

from vericcl.errors import SemanticError
from vericcl.verification.online.statistics import (
    PerformanceHistory,
    summarize_runs,
)


pytestmark = pytest.mark.phase06


def test_statistics_use_median_nearest_rank_p95_and_population_stddev():
    samples = tuple(float(value) for value in range(1, 21))

    result = summarize_runs(samples)

    assert result.samples_us == samples
    assert result.sample_count == 20
    assert result.median_us == pytest.approx(10.5)
    assert result.p95_us == pytest.approx(19.0)
    assert result.mean_us == pytest.approx(10.5)
    assert result.population_standard_deviation_us == pytest.approx(
        math.sqrt(33.25)
    )
    assert result.coefficient_of_variation == pytest.approx(
        math.sqrt(33.25) / 10.5
    )
    assert result.stable is False


def test_statistics_require_exactly_twenty_positive_finite_samples():
    with pytest.raises(SemanticError, match="exactly 20"):
        summarize_runs(tuple(float(value) for value in range(1, 20)))
    for invalid in (0.0, -1.0, math.inf, math.nan, True):
        samples = [1.0] * 20
        samples[-1] = invalid
        with pytest.raises(SemanticError, match="sample"):
            summarize_runs(samples)


def test_retry_history_retains_every_round_and_stops_after_three():
    stable_samples = tuple(100.0 + (index % 2) for index in range(20))
    unstable_samples = (100.0,) * 19 + (300.0,)
    history = PerformanceHistory()

    history = history.add_round(unstable_samples)
    assert history.retry_required is True
    assert history.stable is False
    history = history.add_round(unstable_samples)
    assert history.retry_required is True
    history = history.add_round(unstable_samples)
    assert history.retry_required is False
    assert history.stable is False
    assert history.all_samples_us == unstable_samples * 3
    with pytest.raises(SemanticError, match="three rounds"):
        history.add_round(stable_samples)

    recovered = PerformanceHistory().add_round(unstable_samples)
    recovered = recovered.add_round(stable_samples)
    assert recovered.retry_required is False
    assert recovered.stable is True
    assert len(recovered.rounds) == 2


def test_stability_threshold_is_inclusive_at_five_percent():
    samples = (95.0,) * 10 + (105.0,) * 10

    result = summarize_runs(samples)

    assert result.mean_us == pytest.approx(100.0)
    assert result.coefficient_of_variation == pytest.approx(0.05)
    assert result.stable is True


def test_history_and_sample_collection_reject_invalid_boundaries():
    with pytest.raises(SemanticError, match="sequence"):
        summarize_runs(None)
    with pytest.raises(SemanticError, match="rounds"):
        PerformanceHistory((object(),))

    stable = summarize_runs((100.0,) * 20)
    unstable = summarize_runs((100.0,) * 19 + (300.0,))
    with pytest.raises(SemanticError, match="exceeds three"):
        PerformanceHistory((unstable,) * 4)
    with pytest.raises(SemanticError, match="stable round"):
        PerformanceHistory((stable, unstable))
    with pytest.raises(SemanticError, match="already stable"):
        PerformanceHistory((stable,)).add_round((100.0,) * 20)

    with pytest.raises(SemanticError, match="sample"):
        replace(stable, samples_us=(object(),), sample_count=1)
