import pytest

from vericcl.errors import InputValidationError, SemanticError
from vericcl.topology.model import DirectedLink, LinkKey, PerformanceCurve
from vericcl.topology.performance import (
    normalize_performance_curve,
    safe_per_channel_bandwidth,
    transfer_duration_us,
)


pytestmark = pytest.mark.phase02


def link(alpha_us=2.0, invbw_us=5.0, bandwidth=None, max_channels=4):
    return DirectedLink(
        key=LinkKey(0, 1),
        max_channels=max_channels,
        performance=PerformanceCurve(
            alpha_us=alpha_us,
            invbw_us=invbw_us,
            bandwidth_bytes_per_us=bandwidth or {},
        ),
        resource_ids=(),
    )


def test_uncalibrated_duration_is_conservative():
    edge = link(alpha_us=2.0, invbw_us=5.0)

    assert transfer_duration_us(edge, 1024, concurrency=3) == 11.0


def test_calibrated_curve_uses_prefix_minimum():
    curve = PerformanceCurve(
        alpha_us=2.0,
        invbw_us=5.0,
        bandwidth_bytes_per_us={1: 100.0, 2: 170.0},
    )

    assert safe_per_channel_bandwidth(curve, 2) == 85.0


def test_calibrated_duration_uses_slice_size_and_safe_bandwidth():
    edge = link(bandwidth={1: 100.0, 2: 170.0})

    assert transfer_duration_us(edge, 850, concurrency=2) == 12.0


def test_calibrated_curve_rejects_missing_intermediate_measurement():
    curve = PerformanceCurve(
        alpha_us=2.0,
        invbw_us=5.0,
        bandwidth_bytes_per_us={1: 100.0, 3: 210.0},
    )

    with pytest.raises(InputValidationError, match="missing calibration point"):
        safe_per_channel_bandwidth(curve, 3)


def test_inconsistent_parameters_keep_invbw_and_emit_warning():
    curve, warnings = normalize_performance_curve(
        alpha_us=2.0,
        beta_us=4.0,
        invbw_us=5.0,
        bandwidth_bytes_per_us={},
    )

    assert curve.invbw_us == 5.0
    assert curve.beta_effective_us == 3.0
    assert warnings == (
        "performance parameters disagree; invbw is authoritative",
    )


def test_consistent_parameters_emit_no_warning():
    curve, warnings = normalize_performance_curve(
        alpha_us=2.0,
        beta_us=3.0,
        invbw_us=5.0,
        bandwidth_bytes_per_us={},
    )

    assert curve.beta_effective_us == 3.0
    assert warnings == ()


def test_duration_rejects_concurrency_above_link_limit():
    edge = link(max_channels=2)

    with pytest.raises(SemanticError, match="max_channels"):
        transfer_duration_us(edge, 1024, concurrency=3)
