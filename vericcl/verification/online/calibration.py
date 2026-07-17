from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from vericcl.errors import SemanticError
from vericcl.topology.model import PerformanceCurve
from vericcl.verification.online.model import PerformanceStatistics


BENCHMARK_SIZE_BYTES = 128 * 1024 * 1024
MAX_CALIBRATION_CONCURRENCY = 32
CALIBRATION_LINK_CLASSES = frozenset({"intra_node", "inter_node"})


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SemanticError("{} must be a non-empty string".format(field))
    return value


def _integer(value: object, field: str, *, minimum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise SemanticError(
            "{} must be an integer of at least {}".format(field, minimum)
        )
    return value


def _number(value: object, field: str, *, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticError("{} must be a number".format(field))
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise SemanticError(
            "{} must be finite and at least {}".format(field, minimum)
        )
    return result


@dataclass(frozen=True)
class CalibrationRequest:
    link_class: str
    slice_size_bytes: int
    max_calibration_channels: int
    datatype: str

    def __post_init__(self) -> None:
        link_class = _identifier(
            self.link_class,
            "calibration_request.link_class",
        )
        if link_class not in CALIBRATION_LINK_CLASSES:
            raise SemanticError("calibration request link class is invalid")
        _integer(
            self.slice_size_bytes,
            "calibration_request.slice_size_bytes",
            minimum=1,
        )
        _integer(
            self.max_calibration_channels,
            "calibration_request.max_calibration_channels",
            minimum=1,
        )
        _identifier(self.datatype, "calibration_request.datatype")

    @property
    def benchmark_size_bytes(self) -> int:
        return BENCHMARK_SIZE_BYTES

    @property
    def benchmark_slice_count(self) -> Optional[int]:
        if self.benchmark_size_bytes % self.slice_size_bytes:
            return None
        return self.benchmark_size_bytes // self.slice_size_bytes


@dataclass(frozen=True)
class CalibrationPoint:
    concurrency: int
    duration_statistics: PerformanceStatistics
    full_wave_count: int
    tail_transfer_count: int

    def __post_init__(self) -> None:
        _integer(
            self.concurrency,
            "calibration_point.concurrency",
            minimum=1,
        )
        if not isinstance(self.duration_statistics, PerformanceStatistics):
            raise SemanticError(
                "calibration point duration_statistics is invalid"
            )
        _integer(
            self.full_wave_count,
            "calibration_point.full_wave_count",
            minimum=1,
        )
        _integer(
            self.tail_transfer_count,
            "calibration_point.tail_transfer_count",
            minimum=0,
        )
        if self.tail_transfer_count >= self.concurrency:
            raise SemanticError(
                "calibration point tail must be smaller than concurrency"
            )

    @property
    def safe_duration_us(self) -> float:
        return self.duration_statistics.p95_us

    @property
    def stable(self) -> bool:
        return self.duration_statistics.stable


@dataclass(frozen=True)
class CalibrationResult:
    request: CalibrationRequest
    points: Tuple[CalibrationPoint, ...]
    curve: Optional[PerformanceCurve]
    skipped_reason: Optional[str]

    def __post_init__(self) -> None:
        if not isinstance(self.request, CalibrationRequest):
            raise SemanticError("calibration result request is invalid")
        points = tuple(self.points)
        if not all(isinstance(point, CalibrationPoint) for point in points):
            raise SemanticError("calibration result points are invalid")
        object.__setattr__(self, "points", points)
        if self.skipped_reason is None:
            if not points or not isinstance(self.curve, PerformanceCurve):
                raise SemanticError(
                    "completed calibration result requires points and curve"
                )
        else:
            _identifier(
                self.skipped_reason,
                "calibration_result.skipped_reason",
            )
            if points or self.curve is not None:
                raise SemanticError(
                    "skipped calibration result must not contain data"
                )

    @property
    def stable(self) -> bool:
        return (
            self.skipped_reason is None
            and bool(self.points)
            and all(point.stable for point in self.points)
        )


def derive_calibrated_curve(
    alpha_us: float,
    slice_size_bytes: int,
    points: Sequence[CalibrationPoint],
) -> PerformanceCurve:
    alpha = _number(alpha_us, "alpha_us", minimum=0.0)
    slice_size = _integer(
        slice_size_bytes,
        "slice_size_bytes",
        minimum=1,
    )
    try:
        ordered = tuple(sorted(points, key=lambda point: point.concurrency))
    except (TypeError, AttributeError) as error:
        raise SemanticError("calibration points are invalid") from error
    if not ordered or not all(
        isinstance(point, CalibrationPoint) for point in ordered
    ):
        raise SemanticError("at least one calibration point is required")
    expected = tuple(range(1, len(ordered) + 1))
    actual = tuple(point.concurrency for point in ordered)
    if actual != expected:
        raise SemanticError(
            "calibration point concurrency must be contiguous from one"
        )
    if not all(point.stable for point in ordered):
        raise SemanticError("calibration points must be stable")
    bandwidth = {}
    for point in ordered:
        duration = point.safe_duration_us
        if duration <= alpha:
            raise SemanticError(
                "calibration duration must be above alpha"
            )
        bandwidth[point.concurrency] = (
            point.concurrency * slice_size / (duration - alpha)
        )
    return PerformanceCurve(
        alpha_us=alpha,
        invbw_us=ordered[0].safe_duration_us,
        bandwidth_bytes_per_us=bandwidth,
    )
