"""Directed topology, shared-resource, and performance models."""

from vericcl.topology.model import (
    DirectedLink,
    LaneKey,
    LinkKey,
    PerformanceCurve,
    SharedResource,
    Topology,
)
from vericcl.topology.legacy import convert_legacy_topology
from vericcl.topology.loader import load_topology, topology_from_mapping
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
    "convert_legacy_topology",
    "load_topology",
    "normalize_performance_curve",
    "safe_per_channel_bandwidth",
    "transfer_duration_us",
    "topology_from_mapping",
]
