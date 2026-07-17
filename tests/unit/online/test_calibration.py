from dataclasses import replace

import pytest

from vericcl.errors import SemanticError
from vericcl.verification.online.calibration import (
    CalibrationPoint,
    CalibrationRequest,
    CalibrationResult,
    derive_calibrated_curve,
)
from vericcl.verification.online.statistics import summarize_runs


pytestmark = pytest.mark.phase06


def _point(concurrency, duration_us, *, stable=True):
    samples = (
        (duration_us,) * 20
        if stable
        else (duration_us,) * 19 + (duration_us * 3.0,)
    )
    return CalibrationPoint(
        concurrency=concurrency,
        duration_statistics=summarize_runs(samples),
        full_wave_count=4,
        tail_transfer_count=0,
    )


def test_calibration_curve_preserves_alpha_and_uses_safe_p95_formula():
    curve = derive_calibrated_curve(
        alpha_us=2.0,
        slice_size_bytes=1000,
        points=(
            _point(1, 12.0),
            _point(2, 8.0),
            _point(3, 7.0),
        ),
    )

    assert curve.alpha_us == pytest.approx(2.0)
    assert curve.invbw_us == pytest.approx(12.0)
    assert curve.beta_effective_us == pytest.approx(10.0)
    assert curve.bandwidth_bytes_per_us[1] == pytest.approx(100.0)
    assert curve.bandwidth_bytes_per_us[2] == pytest.approx(2000.0 / 6.0)
    assert curve.bandwidth_bytes_per_us[3] == pytest.approx(600.0)


def test_curve_rejects_invalid_incomplete_or_unstable_points():
    with pytest.raises(SemanticError, match="above alpha"):
        derive_calibrated_curve(2.0, 1000, (_point(1, 2.0),))
    with pytest.raises(SemanticError, match="contiguous"):
        derive_calibrated_curve(
            2.0,
            1000,
            (_point(1, 12.0), _point(3, 7.0)),
        )
    with pytest.raises(SemanticError, match="stable"):
        derive_calibrated_curve(
            2.0,
            1000,
            (_point(1, 12.0, stable=False),),
        )
    with pytest.raises(SemanticError, match="point"):
        derive_calibrated_curve(2.0, 1000, ())


def test_calibration_models_reject_inconsistent_boundaries():
    request = CalibrationRequest("intra_node", 1024, 4, "float")
    for changes in (
        {"link_class": "invalid"},
        {"slice_size_bytes": 0},
        {"max_calibration_channels": 0},
        {"datatype": ""},
    ):
        with pytest.raises(SemanticError):
            replace(request, **changes)

    point = _point(1, 12.0)
    for changes in (
        {"concurrency": 0},
        {"duration_statistics": object()},
        {"full_wave_count": 0},
        {"tail_transfer_count": -1},
    ):
        with pytest.raises(SemanticError):
            replace(point, **changes)

    curve = derive_calibrated_curve(2.0, 1000, (point,))
    valid = CalibrationResult(
        request=request,
        points=(point,),
        curve=curve,
        skipped_reason=None,
    )
    assert valid.stable is True
    with pytest.raises(SemanticError, match="request"):
        replace(valid, request=object())
    with pytest.raises(SemanticError, match="curve"):
        replace(valid, curve=None)

    skipped = CalibrationResult(
        request=request,
        points=(),
        curve=None,
        skipped_reason="slice_size_not_divisible",
    )
    assert skipped.stable is False
    with pytest.raises(SemanticError, match="skipped"):
        replace(skipped, points=(point,))
