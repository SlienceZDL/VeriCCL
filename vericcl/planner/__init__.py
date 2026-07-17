"""Logical collective planning and communication-group discovery."""

from vericcl.planner.build import build_plan
from vericcl.planner.direct import (
    build_direct_plan,
    build_internal_gather,
    build_internal_scatter,
)
from vericcl.planner.dual import DualTree, extract_dual_trees
from vericcl.planner.groups import (
    CommunicationGroups,
    discover_communication_groups,
)
from vericcl.planner.hierarchy import validate_manual_hierarchy
from vericcl.planner.model import (
    LogicalValue,
    PlanDAG,
    PlanEdge,
    PlanNode,
    StageInterface,
)

__all__ = [
    "CommunicationGroups",
    "DualTree",
    "LogicalValue",
    "PlanDAG",
    "PlanEdge",
    "PlanNode",
    "StageInterface",
    "build_direct_plan",
    "build_plan",
    "build_internal_gather",
    "build_internal_scatter",
    "discover_communication_groups",
    "extract_dual_trees",
    "validate_manual_hierarchy",
]
