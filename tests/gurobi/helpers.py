from dataclasses import replace
from pathlib import Path

import pytest

from vericcl.input.loader import resolve_inputs
from vericcl.input.models import AtomConstraints
from vericcl.planner.build import build_plan
from vericcl.planner.model import PlanNode, StageInterface
from vericcl.semantics.collective import (
    CollectiveKind,
    CollectiveSpec,
    OutputSlot,
)
from vericcl.solver.demands import build_solver_problem
from vericcl.solver.gurobi_api import GurobiAdapter
from vericcl.topology.loader import load_topology
from vericcl.topology.model import (
    DirectedLink,
    LinkKey,
    PerformanceCurve,
    SharedResource,
    Topology,
)


EXAMPLES = Path(__file__).parents[2] / "vericcl" / "examples"


def require_gurobi_license():
    if not GurobiAdapter.available():
        pytest.skip("gurobipy is unavailable")
    gp = GurobiAdapter.require()
    try:
        model = gp.Model("vericcl-license-probe")
        model.Params.OutputFlag = 0
        model.dispose()
    except gp.GurobiError as error:
        pytest.skip("Gurobi license is unavailable: {}".format(error))


def broadcast_problem(logical_positions=(0, 1), forbidden=()):
    inputs = resolve_inputs(
        EXAMPLES / "topo" / "two_rank.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    inputs = replace(
        inputs,
        atom_constraints=AtomConstraints(
            stage_num=None,
            forbidden_transfers=tuple(forbidden),
        ),
    )
    topology = load_topology(inputs)
    logical_input = {}
    logical_output = {}
    for logical_position in logical_positions:
        contributors = frozenset({logical_position})
        logical_input[OutputSlot(0, logical_position)] = contributors
        logical_output[OutputSlot(0, logical_position)] = contributors
        logical_output[OutputSlot(1, logical_position)] = contributors
    node = PlanNode(
        node_id="milp-broadcast",
        stage_id=0,
        local_collective=CollectiveSpec(
            kind=CollectiveKind.BROADCAST,
            datatype="float32",
            root=0,
        ),
        communication_group=(0, 1),
        logical_input=StageInterface(logical_input),
        logical_output=StageInterface(logical_output),
        allowed_links=frozenset(topology.links),
        shared_resource_ids=frozenset(),
    )
    return build_solver_problem(node, inputs, topology)


def reduction_dual_problem():
    inputs = resolve_inputs(
        EXAMPLES / "topo" / "two_rank.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    topology = load_topology(inputs)
    plan = build_plan(inputs, topology)
    node = next(
        item
        for item in plan.nodes
        if item.node_id == "allreduce-rs-a00000000"
    )
    return build_solver_problem(node, inputs, topology)


def multihop_problem(shared_resource=False):
    inputs = resolve_inputs(
        EXAMPLES / "topo" / "two_rank.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    inputs = replace(inputs, rank_count=3)
    curve = PerformanceCurve(1.0, 2.0, {})
    keys = (LinkKey(0, 1), LinkKey(1, 2))
    resource_ids = ("shared-path",) if shared_resource else ()
    links = {
        key: DirectedLink(
            key=key,
            max_channels=2,
            performance=curve,
            resource_ids=resource_ids,
        )
        for key in keys
    }
    resources = {}
    if shared_resource:
        resource = SharedResource(
            resource_id="shared-path",
            member_links=keys,
            max_channels=1,
            performance=curve,
        )
        resources[resource.resource_id] = resource
    topology = Topology(
        rank_count=3,
        links=links,
        shared_resources=resources,
        node_membership={0: 0, 1: 0, 2: 0},
        gateways=frozenset(),
        warnings=(),
    )
    contributors = frozenset({0})
    node = PlanNode(
        node_id="milp-multihop",
        stage_id=0,
        local_collective=CollectiveSpec(
            kind=CollectiveKind.BROADCAST,
            datatype="float32",
            root=0,
        ),
        communication_group=(0, 1, 2),
        logical_input=StageInterface({OutputSlot(0, 0): contributors}),
        logical_output=StageInterface(
            {
                OutputSlot(0, 0): contributors,
                OutputSlot(2, 0): contributors,
            }
        ),
        allowed_links=frozenset(keys),
        shared_resource_ids=frozenset(resources),
    )
    return build_solver_problem(node, inputs, topology)


def batching_problem():
    inputs = resolve_inputs(
        EXAMPLES / "topo" / "two_rank.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    inputs = replace(
        inputs,
        rank_count=3,
        strategies=replace(inputs.strategies, batching=True),
    )
    direct_curve = PerformanceCurve(0.0, 4.0, {})
    alternate_curve = PerformanceCurve(0.0, 1.25, {})
    links = {}
    for key, channels, curve in (
        (LinkKey(0, 2), 1, direct_curve),
        (LinkKey(0, 1), 2, alternate_curve),
        (LinkKey(1, 2), 2, alternate_curve),
    ):
        links[key] = DirectedLink(
            key=key,
            max_channels=channels,
            performance=curve,
            resource_ids=(),
        )
    topology = Topology(
        rank_count=3,
        links=links,
        shared_resources={},
        node_membership={0: 0, 1: 0, 2: 0},
        gateways=frozenset(),
        warnings=(),
    )
    logical_input = {}
    logical_output = {}
    for logical_position in (0, 1):
        contributors = frozenset({logical_position})
        logical_input[OutputSlot(0, logical_position)] = contributors
        for rank in range(3):
            logical_output[OutputSlot(rank, logical_position)] = contributors
    node = PlanNode(
        node_id="milp-batching",
        stage_id=0,
        local_collective=CollectiveSpec(
            kind=CollectiveKind.BROADCAST,
            datatype="float32",
            root=0,
        ),
        communication_group=(0, 1, 2),
        logical_input=StageInterface(logical_input),
        logical_output=StageInterface(logical_output),
        allowed_links=frozenset(links),
        shared_resource_ids=frozenset(),
    )
    return build_solver_problem(node, inputs, topology)


def zero_duration_cycle_problem():
    inputs = resolve_inputs(
        EXAMPLES / "topo" / "two_rank.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    inputs = replace(inputs, rank_count=4)
    curve = PerformanceCurve(0.0, 0.0, {})
    keys = (
        LinkKey(0, 1),
        LinkKey(0, 2),
        LinkKey(2, 3),
        LinkKey(3, 2),
        LinkKey(3, 1),
    )
    links = {
        key: DirectedLink(
            key=key,
            max_channels=1,
            performance=curve,
            resource_ids=(),
        )
        for key in keys
    }
    topology = Topology(
        rank_count=4,
        links=links,
        shared_resources={},
        node_membership={rank: 0 for rank in range(4)},
        gateways=frozenset(),
        warnings=(),
    )
    contributors = frozenset({0})
    node = PlanNode(
        node_id="milp-zero-cycle",
        stage_id=0,
        local_collective=CollectiveSpec(
            kind=CollectiveKind.BROADCAST,
            datatype="float32",
            root=0,
        ),
        communication_group=(0, 1, 2, 3),
        logical_input=StageInterface({OutputSlot(0, 0): contributors}),
        logical_output=StageInterface(
            {
                OutputSlot(0, 0): contributors,
                OutputSlot(1, 0): contributors,
            }
        ),
        allowed_links=frozenset(keys),
        shared_resource_ids=frozenset(),
    )
    return build_solver_problem(node, inputs, topology)
