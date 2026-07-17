from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

from vericcl.errors import SemanticError


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticError("{} must be a number".format(field))
    result = float(value)
    if not math.isfinite(result):
        raise SemanticError("{} must be finite".format(field))
    return result


def _rank(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SemanticError("{} must be a non-negative integer".format(field))
    return value


class ClockOrdering(str, Enum):
    BEFORE = "before"
    AFTER = "after"
    UNORDERED = "unordered"


@dataclass(frozen=True)
class AlignedTimestamp:
    value_us: float
    uncertainty_us: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "value_us",
            _finite(self.value_us, "aligned_timestamp.value_us"),
        )
        uncertainty = _finite(
            self.uncertainty_us,
            "aligned_timestamp.uncertainty_us",
        )
        if uncertainty < 0.0:
            raise SemanticError("timestamp uncertainty must be non-negative")
        object.__setattr__(self, "uncertainty_us", uncertainty)


@dataclass(frozen=True)
class ClockSyncSample:
    rank: int
    gpu_ticks: int
    host_before_us: float
    host_after_us: float
    reference_offset_us: float
    reference_uncertainty_us: float

    def __post_init__(self) -> None:
        _rank(self.rank, "clock_sample.rank")
        if (
            isinstance(self.gpu_ticks, bool)
            or not isinstance(self.gpu_ticks, int)
            or self.gpu_ticks < 0
        ):
            raise SemanticError("clock_sample.gpu_ticks must be non-negative")
        before = _finite(
            self.host_before_us,
            "clock_sample.host_before_us",
        )
        after = _finite(
            self.host_after_us,
            "clock_sample.host_after_us",
        )
        if before > after:
            raise SemanticError("clock sample host interval is reversed")
        object.__setattr__(self, "host_before_us", before)
        object.__setattr__(self, "host_after_us", after)
        object.__setattr__(
            self,
            "reference_offset_us",
            _finite(
                self.reference_offset_us,
                "clock_sample.reference_offset_us",
            ),
        )
        uncertainty = _finite(
            self.reference_uncertainty_us,
            "clock_sample.reference_uncertainty_us",
        )
        if uncertainty < 0.0:
            raise SemanticError(
                "clock sample reference uncertainty must be non-negative"
            )
        object.__setattr__(self, "reference_uncertainty_us", uncertainty)

    @property
    def reference_midpoint_us(self) -> float:
        return (
            (self.host_before_us + self.host_after_us) / 2.0
            + self.reference_offset_us
        )

    @property
    def sample_uncertainty_us(self) -> float:
        return (
            (self.host_after_us - self.host_before_us) / 2.0
            + self.reference_uncertainty_us
        )


@dataclass(frozen=True)
class ClockTransform:
    rank: int
    slope_us_per_tick: float
    intercept_us: float
    uncertainty_us: float
    sample_count: int
    origin_ticks: int = 0
    origin_time_us: Optional[float] = None

    def __post_init__(self) -> None:
        _rank(self.rank, "clock_transform.rank")
        slope = _finite(
            self.slope_us_per_tick,
            "clock_transform.slope_us_per_tick",
        )
        if slope <= 0.0:
            raise SemanticError("clock transform slope must be positive")
        object.__setattr__(self, "slope_us_per_tick", slope)
        object.__setattr__(
            self,
            "intercept_us",
            _finite(self.intercept_us, "clock_transform.intercept_us"),
        )
        uncertainty = _finite(
            self.uncertainty_us,
            "clock_transform.uncertainty_us",
        )
        if uncertainty < 0.0:
            raise SemanticError("clock transform uncertainty must be non-negative")
        object.__setattr__(self, "uncertainty_us", uncertainty)
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 2
        ):
            raise SemanticError("clock transform needs at least two samples")
        if (
            isinstance(self.origin_ticks, bool)
            or not isinstance(self.origin_ticks, int)
            or self.origin_ticks < 0
        ):
            raise SemanticError("clock transform origin ticks are invalid")
        origin_time = (
            self.intercept_us
            if self.origin_time_us is None
            else _finite(
                self.origin_time_us,
                "clock_transform.origin_time_us",
            )
        )
        object.__setattr__(self, "origin_time_us", origin_time)

    def timestamp(self, gpu_ticks: int) -> AlignedTimestamp:
        if (
            isinstance(gpu_ticks, bool)
            or not isinstance(gpu_ticks, int)
            or gpu_ticks < 0
        ):
            raise SemanticError("GPU timestamp must be non-negative")
        return AlignedTimestamp(
            self.slope_us_per_tick * (gpu_ticks - self.origin_ticks)
            + self.origin_time_us,
            self.uncertainty_us,
        )


@dataclass(frozen=True)
class ClockAlignment:
    transforms: Mapping[int, ClockTransform]

    def __post_init__(self) -> None:
        try:
            transforms = dict(self.transforms)
        except (TypeError, ValueError) as error:
            raise SemanticError("clock transforms must be a mapping") from error
        if not transforms:
            raise SemanticError("clock alignment must contain transforms")
        for rank, transform in transforms.items():
            if not isinstance(transform, ClockTransform) or transform.rank != rank:
                raise SemanticError("clock transform rank is inconsistent")
        object.__setattr__(self, "transforms", MappingProxyType(transforms))

    def timestamp(self, rank: int, gpu_ticks: int) -> AlignedTimestamp:
        try:
            transform = self.transforms[rank]
        except KeyError as error:
            raise SemanticError("clock transform is missing for rank") from error
        return transform.timestamp(gpu_ticks)

    @staticmethod
    def compare_timestamps(
        left: AlignedTimestamp,
        right: AlignedTimestamp,
    ) -> ClockOrdering:
        if not isinstance(left, AlignedTimestamp) or not isinstance(
            right,
            AlignedTimestamp,
        ):
            raise SemanticError("clock comparison requires aligned timestamps")
        uncertainty = left.uncertainty_us + right.uncertainty_us
        difference = right.value_us - left.value_us
        if difference > uncertainty:
            return ClockOrdering.BEFORE
        if difference < -uncertainty:
            return ClockOrdering.AFTER
        return ClockOrdering.UNORDERED

    def compare(
        self,
        left_rank: int,
        left_ticks: int,
        right_rank: int,
        right_ticks: int,
    ) -> ClockOrdering:
        return self.compare_timestamps(
            self.timestamp(left_rank, left_ticks),
            self.timestamp(right_rank, right_ticks),
        )


def _fit_transform(
    rank: int,
    samples: Sequence[ClockSyncSample],
) -> ClockTransform:
    if len(samples) < 2:
        raise SemanticError(
            "clock sync rank {} needs at least two samples".format(rank)
        )
    origin_ticks = samples[0].gpu_ticks
    xs = tuple(float(sample.gpu_ticks - origin_ticks) for sample in samples)
    ys = tuple(sample.reference_midpoint_us for sample in samples)
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    variance = sum((value - mean_x) ** 2 for value in xs)
    if variance <= 0.0:
        raise SemanticError(
            "clock sync rank {} needs distinct GPU ticks".format(rank)
        )
    slope = sum(
        (left - mean_x) * (right - mean_y)
        for left, right in zip(xs, ys)
    ) / variance
    if slope <= 0.0:
        raise SemanticError("clock sync fitted a non-positive slope")
    origin_time = mean_y - slope * mean_x
    intercept = origin_time - slope * origin_ticks
    uncertainty = max(
        abs((slope * x + origin_time) - y) + sample.sample_uncertainty_us
        for x, y, sample in zip(xs, ys, samples)
    )
    return ClockTransform(
        rank,
        slope,
        intercept,
        uncertainty,
        len(samples),
        origin_ticks,
        origin_time,
    )


def align_clocks(
    records_by_rank: Mapping[int, Sequence[object]],
    samples: Sequence[ClockSyncSample],
) -> ClockAlignment:
    if not isinstance(records_by_rank, Mapping) or not records_by_rank:
        raise SemanticError("records_by_rank must be a non-empty mapping")
    ranks = set()
    for rank, records in records_by_rank.items():
        _rank(rank, "records_by_rank.rank")
        ranks.add(rank)
        try:
            normalized = tuple(records)
        except TypeError as error:
            raise SemanticError("rank records must be iterable") from error
        if any(getattr(record, "rank", rank) != rank for record in normalized):
            raise SemanticError("record rank differs from mapping key")
    if not isinstance(samples, Sequence):
        raise SemanticError("clock samples must be a sequence")
    grouped = {rank: [] for rank in ranks}
    for sample in samples:
        if not isinstance(sample, ClockSyncSample):
            raise SemanticError("clock sample is invalid")
        if sample.rank in grouped:
            grouped[sample.rank].append(sample)
    transforms = {
        rank: _fit_transform(rank, tuple(grouped[rank]))
        for rank in sorted(ranks)
    }
    return ClockAlignment(transforms)


def parse_clock_sync_output(text: str) -> Tuple[ClockSyncSample, ...]:
    if not isinstance(text, str):
        raise SemanticError("clock sync output must be text")
    result = []
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] != "VERICCL_CLOCK_SYNC":
            continue
        if len(fields) != 7:
            raise SemanticError("clock sync record has an invalid field count")
        try:
            rank = int(fields[1])
            gpu_ticks = int(fields[2])
            values_us = tuple(int(value) / 1000.0 for value in fields[3:])
        except ValueError as error:
            raise SemanticError("clock sync record is invalid") from error
        result.append(
            ClockSyncSample(rank, gpu_ticks, *values_us)
        )
    if not result:
        raise SemanticError("clock sync output contains no samples")
    return tuple(result)
