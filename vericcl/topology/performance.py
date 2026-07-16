import math
from typing import Mapping, Optional, Tuple

from vericcl.errors import InputValidationError, SemanticError
from vericcl.topology.model import DirectedLink, PerformanceCurve


def _number(value: object, field: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputValidationError("{} must be a number".format(field))
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < minimum:
        raise InputValidationError(
            "{} must be finite and at least {}".format(field, minimum)
        )
    return normalized


def _concurrency(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SemanticError("concurrency must be a positive integer")
    return value


def normalize_performance_curve(
    *,
    alpha_us: object,
    beta_us: Optional[object],
    invbw_us: Optional[object],
    bandwidth_bytes_per_us: Mapping[int, float],
) -> Tuple[PerformanceCurve, Tuple[str, ...]]:
    alpha = _number(alpha_us, "alpha_us")
    if beta_us is None and invbw_us is None:
        raise InputValidationError("beta_us or invbw_us is required")
    beta = None if beta_us is None else _number(beta_us, "beta_us")
    invbw = None if invbw_us is None else _number(invbw_us, "invbw_us")
    warnings = ()
    if invbw is None:
        invbw = alpha + beta
    elif beta is not None and not math.isclose(
        invbw,
        alpha + beta,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        warnings = (
            "performance parameters disagree; invbw is authoritative",
        )
    if invbw < alpha:
        raise InputValidationError("invbw_us must not be below alpha_us")
    curve = PerformanceCurve(
        alpha_us=alpha,
        invbw_us=invbw,
        bandwidth_bytes_per_us=bandwidth_bytes_per_us,
    )
    return curve, warnings


def safe_per_channel_bandwidth(
    curve: PerformanceCurve,
    concurrency: int,
) -> float:
    if not isinstance(curve, PerformanceCurve):
        raise SemanticError("curve must be a PerformanceCurve")
    normalized_concurrency = _concurrency(concurrency)
    if not curve.bandwidth_bytes_per_us:
        raise InputValidationError("calibrated bandwidth points are unavailable")
    missing = [
        value
        for value in range(1, normalized_concurrency + 1)
        if value not in curve.bandwidth_bytes_per_us
    ]
    if missing:
        raise InputValidationError(
            "missing calibration point for concurrency {}".format(missing[0])
        )
    return min(
        curve.bandwidth_bytes_per_us[value] / value
        for value in range(1, normalized_concurrency + 1)
    )


def transfer_duration_us(
    link: DirectedLink,
    slice_size_bytes: int,
    concurrency: int,
) -> float:
    if not isinstance(link, DirectedLink):
        raise SemanticError("link must be a DirectedLink")
    if (
        isinstance(slice_size_bytes, bool)
        or not isinstance(slice_size_bytes, int)
        or slice_size_bytes < 1
    ):
        raise SemanticError("slice_size_bytes must be a positive integer")
    normalized_concurrency = _concurrency(concurrency)
    if normalized_concurrency > link.max_channels:
        raise SemanticError("concurrency exceeds link max_channels")
    curve = link.performance
    if curve.is_calibrated:
        bandwidth = safe_per_channel_bandwidth(curve, normalized_concurrency)
        return curve.alpha_us + slice_size_bytes / bandwidth
    return curve.alpha_us + normalized_concurrency * curve.beta_effective_us
