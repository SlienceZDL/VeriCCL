from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from vericcl.errors import SemanticError
from vericcl.semantics.collective import CollectiveKind


ONLINE_COLLECTIVE_KINDS = frozenset(
    {
        CollectiveKind.BROADCAST,
        CollectiveKind.REDUCE,
        CollectiveKind.ALL_GATHER,
        CollectiveKind.ALL_REDUCE,
        CollectiveKind.ALL_TO_ALL,
        CollectiveKind.REDUCE_SCATTER,
    }
)

_REDUCTION_KINDS = frozenset(
    {
        CollectiveKind.REDUCE,
        CollectiveKind.ALL_REDUCE,
        CollectiveKind.REDUCE_SCATTER,
    }
)

_ROOTED_KINDS = frozenset(
    {
        CollectiveKind.BROADCAST,
        CollectiveKind.REDUCE,
    }
)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise SemanticError("{} must be a non-empty string".format(field))
    if any(character.isspace() for character in value):
        raise SemanticError("{} must not contain whitespace".format(field))
    return value


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SemanticError("{} must be a positive integer".format(field))
    return value


def _nonnegative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SemanticError("{} must be a non-negative integer".format(field))
    return value


def _finite_number(
    value: object,
    field: str,
    *,
    positive: bool,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticError("{} must be a number".format(field))
    result = float(value)
    minimum_valid = result > 0.0 if positive else result >= 0.0
    if not math.isfinite(result) or not minimum_valid:
        qualifier = "positive" if positive else "non-negative"
        raise SemanticError(
            "{} must be finite and {}".format(field, qualifier)
        )
    return result


@dataclass(frozen=True)
class NcclTestRequest:
    kind: CollectiveKind
    message_size_bytes: int
    datatype: str
    reduction_op: Optional[str]
    root: Optional[int]
    inplace: bool
    binary_directory: Optional[str] = None

    def __post_init__(self) -> None:
        try:
            kind = CollectiveKind(self.kind)
        except (TypeError, ValueError) as error:
            raise SemanticError("request kind is invalid") from error
        if kind not in ONLINE_COLLECTIVE_KINDS:
            raise SemanticError("request kind is not an online collective")
        object.__setattr__(self, "kind", kind)
        _positive_integer(
            self.message_size_bytes,
            "nccl_test_request.message_size_bytes",
        )
        _identifier(self.datatype, "nccl_test_request.datatype")
        if not isinstance(self.inplace, bool):
            raise SemanticError("nccl_test_request.inplace must be a boolean")

        if kind in _REDUCTION_KINDS:
            _identifier(
                self.reduction_op,
                "nccl_test_request.reduction_op",
            )
        elif self.reduction_op is not None:
            raise SemanticError(
                "non-reduction collective must not define reduction_op"
            )

        if kind in _ROOTED_KINDS:
            _nonnegative_integer(self.root, "nccl_test_request.root")
        elif self.root is not None:
            raise SemanticError(
                "unrooted collective must not define root"
            )

        if self.binary_directory is not None:
            try:
                directory = str(Path(self.binary_directory))
            except TypeError as error:
                raise SemanticError(
                    "nccl_test_request.binary_directory is invalid"
                ) from error
            if not directory:
                raise SemanticError(
                    "nccl_test_request.binary_directory must be non-empty"
                )
            object.__setattr__(self, "binary_directory", directory)


@dataclass(frozen=True)
class NcclTestMeasurement:
    time_us: float
    algorithm_bandwidth_gbps: float
    bus_bandwidth_gbps: float
    wrong_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "time_us",
            _finite_number(
                self.time_us,
                "nccl_test_measurement.time_us",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "algorithm_bandwidth_gbps",
            _finite_number(
                self.algorithm_bandwidth_gbps,
                "nccl_test_measurement.algorithm_bandwidth_gbps",
                positive=False,
            ),
        )
        object.__setattr__(
            self,
            "bus_bandwidth_gbps",
            _finite_number(
                self.bus_bandwidth_gbps,
                "nccl_test_measurement.bus_bandwidth_gbps",
                positive=False,
            ),
        )
        _nonnegative_integer(
            self.wrong_count,
            "nccl_test_measurement.wrong_count",
        )


@dataclass(frozen=True)
class NcclTestRun:
    message_size_bytes: int
    element_count: int
    datatype: str
    metadata_fields: Tuple[str, ...]
    out_of_place: NcclTestMeasurement
    in_place: Optional[NcclTestMeasurement]

    def __post_init__(self) -> None:
        _positive_integer(
            self.message_size_bytes,
            "nccl_test_run.message_size_bytes",
        )
        _positive_integer(self.element_count, "nccl_test_run.element_count")
        _identifier(self.datatype, "nccl_test_run.datatype")
        metadata = tuple(self.metadata_fields)
        if not all(isinstance(value, str) and value for value in metadata):
            raise SemanticError("nccl_test_run metadata fields are invalid")
        object.__setattr__(self, "metadata_fields", metadata)
        if not isinstance(self.out_of_place, NcclTestMeasurement):
            raise SemanticError("nccl_test_run out_of_place is invalid")
        if self.in_place is not None and not isinstance(
            self.in_place,
            NcclTestMeasurement,
        ):
            raise SemanticError("nccl_test_run in_place is invalid")

    def selected_time_us(self, *, inplace: bool) -> float:
        if not isinstance(inplace, bool):
            raise SemanticError("inplace selector must be a boolean")
        if not inplace:
            return self.out_of_place.time_us
        if self.in_place is None:
            raise SemanticError("in-place performance measurement is missing")
        return self.in_place.time_us


@dataclass(frozen=True)
class PerformanceStatistics:
    samples_us: Tuple[float, ...]
    sample_count: int
    median_us: float
    p95_us: float
    mean_us: float
    population_standard_deviation_us: float
    coefficient_of_variation: float
    stable: bool

    def __post_init__(self) -> None:
        samples = tuple(
            _finite_number(
                value,
                "performance_statistics.sample",
                positive=True,
            )
            for value in self.samples_us
        )
        object.__setattr__(self, "samples_us", samples)
        if self.sample_count != len(samples):
            raise SemanticError(
                "performance statistics sample count does not match"
            )
        for field, positive in (
            ("median_us", True),
            ("p95_us", True),
            ("mean_us", True),
            ("population_standard_deviation_us", False),
            ("coefficient_of_variation", False),
        ):
            object.__setattr__(
                self,
                field,
                _finite_number(
                    getattr(self, field),
                    "performance_statistics.{}".format(field),
                    positive=positive,
                ),
            )
        if not isinstance(self.stable, bool):
            raise SemanticError(
                "performance_statistics.stable must be a boolean"
            )
