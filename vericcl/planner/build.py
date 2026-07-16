from vericcl.errors import InputValidationError
from vericcl.input.models import ResolvedInput
from vericcl.planner.direct import build_direct_plan
from vericcl.planner.groups import discover_communication_groups
from vericcl.planner.hierarchy import (
    build_gateway_allreduce_plan,
    build_manual_plan,
)
from vericcl.planner.model import PlanDAG
from vericcl.semantics.collective import CollectiveKind
from vericcl.topology.model import Topology


def _validate_stage_count(plan: PlanDAG, expected: object) -> None:
    stage_ids = sorted({node.stage_id for node in plan.nodes})
    if stage_ids != list(range(len(stage_ids))):
        raise InputValidationError("plan stage IDs must be contiguous from zero")
    if expected is not None and expected != len(stage_ids):
        raise InputValidationError(
            "atom stage_num does not match the generated plan"
        )


def build_plan(inputs: ResolvedInput, topology: Topology) -> PlanDAG:
    if not isinstance(inputs, ResolvedInput):
        raise InputValidationError("inputs must be a ResolvedInput")
    if not isinstance(topology, Topology):
        raise InputValidationError("topology must be a Topology")
    if inputs.rank_count != topology.rank_count:
        raise InputValidationError("input and topology rank counts do not match")
    if inputs.strategies.manual_hierarchy:
        plan = build_manual_plan(inputs, topology)
    elif (
        inputs.strategies.hierarchy
        and inputs.collective.kind is CollectiveKind.ALL_REDUCE
    ):
        groups = discover_communication_groups(topology)
        if groups.inter_node:
            plan = build_gateway_allreduce_plan(inputs, topology, groups)
        else:
            plan = build_direct_plan(inputs, topology)
    else:
        plan = build_direct_plan(inputs, topology)
    _validate_stage_count(plan, inputs.atom_constraints.stage_num)
    return plan
