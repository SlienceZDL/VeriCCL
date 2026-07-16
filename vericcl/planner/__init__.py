"""Logical collective planning and communication-group discovery."""

from vericcl.planner.direct import (
    build_direct_plan,
    build_internal_gather,
    build_internal_scatter,
)
from vericcl.planner.groups import (
    CommunicationGroups,
    discover_communication_groups,
)
from vericcl.planner.model import (
    LogicalValue,
    PlanDAG,
    PlanEdge,
    PlanNode,
    StageInterface,
)

__all__ = [
    "CommunicationGroups",
    "LogicalValue",
    "PlanDAG",
    "PlanEdge",
    "PlanNode",
    "StageInterface",
    "build_direct_plan",
    "build_internal_gather",
    "build_internal_scatter",
    "discover_communication_groups",
]
