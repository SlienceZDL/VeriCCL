from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Sequence, Tuple

from vericcl.errors import SemanticError
from vericcl.verification.online.model import PerformanceStatistics


MEASUREMENT_SAMPLE_COUNT = 20
MAX_MEASUREMENT_ROUNDS = 3
STABILITY_CV_THRESHOLD = 0.05


def _samples(values: Sequence[float]) -> Tuple[float, ...]:
    try:
        samples = tuple(values)
    except TypeError as error:
        raise SemanticError("performance samples must be a sequence") from error
    if len(samples) != MEASUREMENT_SAMPLE_COUNT:
        raise SemanticError("performance statistics require exactly 20 samples")
    normalized = []
    for value in samples:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SemanticError("performance sample must be a number")
        sample = float(value)
        if not math.isfinite(sample) or sample <= 0.0:
            raise SemanticError(
                "performance sample must be finite and positive"
            )
        normalized.append(sample)
    return tuple(normalized)


def summarize_runs(samples_us: Sequence[float]) -> PerformanceStatistics:
    samples = _samples(samples_us)
    ordered = tuple(sorted(samples))
    p95_rank = math.ceil(0.95 * len(ordered))
    mean_us = statistics.fmean(samples)
    standard_deviation = statistics.pstdev(samples)
    variation = standard_deviation / mean_us
    return PerformanceStatistics(
        samples_us=samples,
        sample_count=len(samples),
        median_us=statistics.median(samples),
        p95_us=ordered[p95_rank - 1],
        mean_us=mean_us,
        population_standard_deviation_us=standard_deviation,
        coefficient_of_variation=variation,
        stable=variation <= STABILITY_CV_THRESHOLD,
    )


@dataclass(frozen=True)
class PerformanceHistory:
    rounds: Tuple[PerformanceStatistics, ...] = ()

    def __post_init__(self) -> None:
        rounds = tuple(self.rounds)
        if not all(isinstance(value, PerformanceStatistics) for value in rounds):
            raise SemanticError("performance history rounds are invalid")
        if len(rounds) > MAX_MEASUREMENT_ROUNDS:
            raise SemanticError("performance history exceeds three rounds")
        if any(value.stable for value in rounds[:-1]):
            raise SemanticError(
                "performance history continued after a stable round"
            )
        object.__setattr__(self, "rounds", rounds)

    @property
    def stable(self) -> bool:
        return bool(self.rounds) and self.rounds[-1].stable

    @property
    def retry_required(self) -> bool:
        return (
            bool(self.rounds)
            and not self.stable
            and len(self.rounds) < MAX_MEASUREMENT_ROUNDS
        )

    @property
    def all_samples_us(self) -> Tuple[float, ...]:
        return tuple(
            sample
            for round_value in self.rounds
            for sample in round_value.samples_us
        )

    def add_round(self, samples_us: Sequence[float]) -> "PerformanceHistory":
        if len(self.rounds) >= MAX_MEASUREMENT_ROUNDS:
            raise SemanticError("performance history already has three rounds")
        if self.stable:
            raise SemanticError("performance history is already stable")
        return PerformanceHistory(self.rounds + (summarize_runs(samples_us),))
