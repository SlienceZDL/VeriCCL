"""Directed topology, shared-resource, and performance models."""

from vericcl.topology.model import (
    DirectedLink,
    LaneKey,
    LinkKey,
    PerformanceCurve,
    SharedResource,
    Topology,
)
from vericcl.topology.performance import (
    normalize_performance_curve,
    safe_per_channel_bandwidth,
    transfer_duration_us,
)

__all__ = [
    "DirectedLink",
    "LaneKey",
    "LinkKey",
    "PerformanceCurve",
    "SharedResource",
    "Topology",
    "normalize_performance_curve",
    "safe_per_channel_bandwidth",
    "transfer_duration_us",
]
