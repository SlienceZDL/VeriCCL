import pytest

from vericcl.errors import SemanticError
from vericcl.verification.online.clock_sync import (
    ClockOrdering,
    ClockSyncSample,
    align_clocks,
    parse_clock_sync_output,
)


pytestmark = pytest.mark.phase06


def _samples():
    return (
        ClockSyncSample(0, 0, -0.1, 0.1, 0.0, 0.0),
        ClockSyncSample(0, 100, 9.9, 10.1, 0.0, 0.0),
        ClockSyncSample(0, 200, 19.9, 20.1, 0.0, 0.0),
        ClockSyncSample(1, 1000, 4.8, 5.2, 5.0, 0.1),
        ClockSyncSample(1, 1100, 14.8, 15.2, 5.0, 0.1),
        ClockSyncSample(1, 1200, 24.8, 25.2, 5.0, 0.1),
    )


def test_affine_fit_aligns_each_rank_to_reference_microseconds():
    alignment = align_clocks({0: (), 1: ()}, _samples())

    assert alignment.transforms[0].slope_us_per_tick == pytest.approx(0.1)
    assert alignment.transforms[0].intercept_us == pytest.approx(0.0)
    assert alignment.transforms[1].slope_us_per_tick == pytest.approx(0.1)
    assert alignment.transforms[1].intercept_us == pytest.approx(-90.0)
    assert alignment.timestamp(0, 100).value_us == pytest.approx(10.0)
    assert alignment.timestamp(1, 1000).value_us == pytest.approx(10.0)
    assert alignment.transforms[1].uncertainty_us >= 0.3


def test_cross_rank_order_below_combined_uncertainty_is_unordered():
    alignment = align_clocks({0: (), 1: ()}, _samples())

    assert alignment.compare(0, 100, 1, 1000) is ClockOrdering.UNORDERED
    assert alignment.compare(0, 100, 1, 1100) is ClockOrdering.BEFORE
    assert alignment.compare(1, 1100, 0, 100) is ClockOrdering.AFTER


def test_affine_fit_is_stable_for_large_gpu_timer_origins():
    origin = 10**18
    samples = (
        ClockSyncSample(0, origin, -0.1, 0.1, 0.0, 0.0),
        ClockSyncSample(0, origin + 100, 9.9, 10.1, 0.0, 0.0),
        ClockSyncSample(0, origin + 200, 19.9, 20.1, 0.0, 0.0),
    )

    alignment = align_clocks({0: ()}, samples)

    assert alignment.timestamp(0, origin + 50).value_us == pytest.approx(5.0)


def test_clock_sync_stdout_parser_converts_nanoseconds_to_microseconds():
    text = "\n".join(
        (
            "noise",
            "VERICCL_CLOCK_SYNC 1 5000 100000 102000 -3000 500",
        )
    )

    assert parse_clock_sync_output(text) == (
        ClockSyncSample(1, 5000, 100.0, 102.0, -3.0, 0.5),
    )


def test_alignment_rejects_missing_or_degenerate_rank_samples():
    with pytest.raises(SemanticError, match="rank 1"):
        align_clocks({0: (), 1: ()}, _samples()[:3])
    with pytest.raises(SemanticError, match="distinct"):
        align_clocks(
            {0: ()},
            (
                ClockSyncSample(0, 1, 0.0, 0.1, 0.0, 0.0),
                ClockSyncSample(0, 1, 1.0, 1.1, 0.0, 0.0),
            ),
        )
