from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional, Sequence, Tuple

from vericcl.constants import SOFTWARE_MAX_CONCURRENCY
from vericcl.errors import SemanticError
from vericcl.topology.model import PerformanceCurve, Topology
from vericcl.verification.online.model import PerformanceStatistics
from vericcl.verification.online.statistics import (
    MEASUREMENT_SAMPLE_COUNT,
    summarize_runs,
)
from vericcl.verification.online.trace_analysis import TraceAnalysis


BENCHMARK_SIZE_BYTES = 128 * 1024 * 1024
MAX_CALIBRATION_CONCURRENCY = SOFTWARE_MAX_CONCURRENCY
CALIBRATION_LINK_CLASSES = frozenset({"intra_node", "inter_node"})
CALIBRATION_TRANSFER_PREFIX = "calibration-send-"


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


def _calibration_logical_index(transfer_id: str) -> int:
    if not isinstance(transfer_id, str) or not transfer_id.startswith(
        CALIBRATION_TRANSFER_PREFIX
    ):
        raise SemanticError("calibration trace contains an unknown transfer")
    suffix = transfer_id[len(CALIBRATION_TRANSFER_PREFIX) :]
    if len(suffix) != 8 or not suffix.isdigit():
        raise SemanticError("calibration trace transfer ID is invalid")
    return int(suffix)


def calibration_point_from_trace(
    request: CalibrationRequest,
    concurrency: int,
    analysis: TraceAnalysis,
) -> CalibrationPoint:
    if not isinstance(request, CalibrationRequest):
        raise SemanticError("calibration trace requires a CalibrationRequest")
    normalized_concurrency = _integer(
        concurrency,
        "calibration trace concurrency",
        minimum=1,
    )
    if not isinstance(analysis, TraceAnalysis):
        raise SemanticError("calibration trace analysis is invalid")
    slice_count = request.benchmark_slice_count
    if slice_count is None:
        raise SemanticError("calibration trace requires divisible slice size")
    full_wave_count = slice_count // normalized_concurrency
    if full_wave_count < 1:
        raise SemanticError("calibration trace contains no full wave")
    intervals = {}
    for interval in analysis.intervals:
        if interval.local is not None:
            continue
        logical = _calibration_logical_index(interval.transfer_id)
        key = (interval.iteration, logical)
        if key in intervals:
            raise SemanticError("calibration trace contains duplicate transfer")
        intervals[key] = interval
    iterations = tuple(sorted({iteration for iteration, _ in intervals}))
    if len(iterations) != MEASUREMENT_SAMPLE_COUNT:
        raise SemanticError("calibration trace requires exactly 20 iterations")
    expected_logical = set(range(slice_count))
    samples = []
    for iteration in iterations:
        actual_logical = {
            logical
            for current_iteration, logical in intervals
            if current_iteration == iteration
        }
        if actual_logical != expected_logical:
            raise SemanticError("calibration trace is incomplete")
        wave_durations = []
        for wave in range(full_wave_count):
            wave_intervals = tuple(
                intervals[(iteration, logical)]
                for logical in range(
                    wave * normalized_concurrency,
                    (wave + 1) * normalized_concurrency,
                )
            )
            duration = max(
                interval.sender_end_us for interval in wave_intervals
            ) - min(
                interval.sender_start_us for interval in wave_intervals
            )
            if duration <= 0.0:
                raise SemanticError("calibration trace wave is not positive")
            wave_durations.append(duration)
        ordered = tuple(sorted(wave_durations))
        p95_rank = math.ceil(0.95 * len(ordered))
        samples.append(ordered[p95_rank - 1])
    return CalibrationPoint(
        concurrency=normalized_concurrency,
        duration_statistics=summarize_runs(samples),
        full_wave_count=full_wave_count,
        tail_transfer_count=slice_count % normalized_concurrency,
    )


def _link_class(topology: Topology, src_rank: int, dst_rank: int) -> str:
    return (
        "intra_node"
        if topology.node_membership[src_rank]
        == topology.node_membership[dst_rank]
        else "inter_node"
    )


def _curve_identity(curve: PerformanceCurve) -> tuple:
    return (
        curve.alpha_us,
        curve.invbw_us,
        tuple(curve.bandwidth_bytes_per_us.items()),
    )


def _calibration_link_identity(topology: Topology, anchor) -> tuple:
    src_node = topology.node_membership[anchor.src_rank]
    dst_node = topology.node_membership[anchor.dst_rank]
    domain_nodes = (
        (src_node,)
        if src_node == dst_node
        else (src_node, dst_node)
    )
    all_nodes = tuple(sorted(set(topology.node_membership.values())))
    node_order = domain_nodes + tuple(
        node for node in all_nodes if node not in domain_nodes
    )
    node_index = {
        node: index for index, node in enumerate(node_order)
    }
    ranks_by_node = {}
    for node in node_order:
        anchors = tuple(
            rank
            for rank in (anchor.src_rank, anchor.dst_rank)
            if topology.node_membership[rank] == node
        )
        remaining = tuple(
            rank
            for rank in range(topology.rank_count)
            if topology.node_membership[rank] == node
            and rank not in anchors
        )
        ranks_by_node[node] = anchors + remaining
    rank_identity = {
        rank: (node_index[node], local_rank)
        for node, ranks in ranks_by_node.items()
        for local_rank, rank in enumerate(ranks)
    }

    def resource_identity(resource_id: str) -> tuple:
        resource = topology.shared_resources[resource_id]
        return (
            resource.max_channels,
            _curve_identity(resource.performance),
            tuple(
                sorted(
                    (
                        rank_identity[member.src_rank],
                        rank_identity[member.dst_rank],
                    )
                    for member in resource.member_links
                )
            ),
        )

    domain_ranks = {
        rank
        for node in domain_nodes
        for rank in ranks_by_node[node]
    }
    links = tuple(
        sorted(
            (
                rank_identity[key.src_rank],
                rank_identity[key.dst_rank],
                edge.max_channels,
                _curve_identity(edge.performance),
                tuple(
                    sorted(
                        resource_identity(resource_id)
                        for resource_id in edge.resource_ids
                    )
                ),
            )
            for key, edge in topology.links.items()
            if key.src_rank in domain_ranks
            and key.dst_rank in domain_ranks
        )
    )
    roles = tuple(
        sorted(
            (
                rank_identity[rank],
                rank in topology.gateways,
            )
            for rank in domain_ranks
        )
    )
    return (
        src_node == dst_node,
        tuple(len(ranks_by_node[node]) for node in domain_nodes),
        rank_identity[anchor.src_rank],
        rank_identity[anchor.dst_rank],
        roles,
        links,
    )


def _curve_for_limit(
    calibration: CalibrationResult,
    alpha_us: float,
    max_channels: int,
) -> PerformanceCurve:
    points = tuple(
        point
        for point in calibration.points
        if point.concurrency <= max_channels
    )
    if not points:
        raise SemanticError("calibration has no supported concurrency point")
    return derive_calibrated_curve(
        alpha_us,
        calibration.request.slice_size_bytes,
        points,
    )


def apply_calibration_to_topology(
    topology: Topology,
    calibration: CalibrationResult,
) -> Topology:
    if not isinstance(topology, Topology):
        raise SemanticError("calibration update requires a Topology")
    if not isinstance(calibration, CalibrationResult):
        raise SemanticError("calibration update requires a CalibrationResult")
    if not calibration.stable:
        raise SemanticError("topology update requires stable calibration")
    link_class = calibration.request.link_class
    matching_links = {
        key
        for key in topology.links
        if _link_class(topology, key.src_rank, key.dst_rank) == link_class
    }
    if not matching_links:
        raise SemanticError("topology has no link in the calibrated class")
    domain_signatures = {
        _calibration_link_identity(topology, key) for key in matching_links
    }
    if len(domain_signatures) != 1:
        raise SemanticError(
            "calibrated link class must contain exactly isomorphic links"
        )
    calibrated_limit = max(
        point.concurrency for point in calibration.points
    )
    links = {
        key: (
            replace(
                edge,
                max_channels=min(edge.max_channels, calibrated_limit),
                performance=_curve_for_limit(
                    calibration,
                    edge.performance.alpha_us,
                    edge.max_channels,
                ),
            )
            if key in matching_links
            else edge
        )
        for key, edge in topology.links.items()
    }
    resources = {}
    for resource_id, resource in topology.shared_resources.items():
        member_classes = {
            _link_class(topology, key.src_rank, key.dst_rank)
            for key in resource.member_links
        }
        resources[resource_id] = (
            replace(
                resource,
                max_channels=min(
                    resource.max_channels,
                    calibrated_limit,
                ),
                performance=_curve_for_limit(
                    calibration,
                    resource.performance.alpha_us,
                    resource.max_channels,
                ),
            )
            if member_classes == {link_class}
            else resource
        )
    return replace(
        topology,
        links=links,
        shared_resources=resources,
        isomorphism_signature="",
    )
