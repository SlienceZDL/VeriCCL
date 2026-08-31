from dataclasses import replace
from pathlib import Path

import pytest

from vericcl.errors import InputValidationError, SemanticError
from vericcl.input.loader import resolve_inputs
from vericcl.planner.direct import (
    build_direct_plan,
    build_internal_gather,
    build_internal_scatter,
)
from vericcl.planner.model import (
    LogicalValue,
    PlanDAG,
    PlanEdge,
    PlanNode,
    PlanningMode,
    StageInterface,
)
from vericcl.semantics.collective import (
    CollectiveKind,
    CollectiveSpec,
    OutputSlot,
    required_outputs,
)
from vericcl.topology.loader import load_topology
from vericcl.topology.model import LinkKey


pytestmark = pytest.mark.phase02


EXAMPLES = Path(__file__).parents[3] / "vericcl" / "examples"


def resolved_input(kind, slice_count=2, hierarchy=False):
    inputs = resolve_inputs(
        EXAMPLES / "topo" / "two_rank.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    rooted = kind in {CollectiveKind.BROADCAST, CollectiveKind.REDUCE}
    reduced = kind in {
        CollectiveKind.REDUCE,
        CollectiveKind.ALL_REDUCE,
        CollectiveKind.REDUCE_SCATTER,
    }
    collective = CollectiveSpec(
        kind=kind,
        datatype="float32",
        reduction_op="sum" if reduced else None,
        root=0 if rooted else None,
        inplace=False,
    )
    return replace(
        inputs,
        collective=collective,
        hyperparameters=replace(
            inputs.hyperparameters,
            total_size_bytes=slice_count,
            slice_size_bytes=1,
        ),
        strategies=replace(inputs.strategies, hierarchy=hierarchy),
    )


def topology_for(inputs):
    return load_topology(inputs)


def test_allgather_plan_is_sum_of_broadcasts():
    inputs = resolved_input(CollectiveKind.ALL_GATHER)

    plan = build_direct_plan(inputs, topology_for(inputs))

    assert [node.local_collective.kind for node in plan.nodes] == [
        CollectiveKind.BROADCAST
    ] * 4
    assert plan.final_outputs.values == required_outputs(
        plan.collective,
        2,
        2,
    )


def test_broadcast_creates_one_node_per_logical_slice():
    inputs = resolved_input(CollectiveKind.BROADCAST)

    plan = build_direct_plan(inputs, topology_for(inputs))

    assert len(plan.nodes) == 2
    assert {node.local_collective.root for node in plan.nodes} == {0}
    assert all(len(node.logical_input.values) == 1 for node in plan.nodes)


@pytest.mark.parametrize(
    "kind",
    [CollectiveKind.REDUCE, CollectiveKind.REDUCE_SCATTER],
)
def test_reduction_plans_create_dual_descriptors(kind):
    inputs = resolved_input(kind)

    plan = build_direct_plan(inputs, topology_for(inputs))

    assert len(plan.nodes) == 2
    assert all(node.local_collective.kind is kind for node in plan.nodes)
    assert all(node.dual_of_node_id is not None for node in plan.nodes)


def test_allreduce_has_reduce_scatter_dependencies_before_allgather():
    inputs = resolved_input(CollectiveKind.ALL_REDUCE)

    plan = build_direct_plan(inputs, topology_for(inputs))

    assert [node.local_collective.kind for node in plan.nodes] == [
        CollectiveKind.REDUCE_SCATTER,
        CollectiveKind.REDUCE_SCATTER,
        CollectiveKind.BROADCAST,
        CollectiveKind.BROADCAST,
    ]
    assert [node.stage_id for node in plan.nodes] == [0, 0, 1, 1]
    assert len(plan.edges) == 2
    assert all(edge.producer_id.startswith("allreduce-rs") for edge in plan.edges)
    assert all(edge.consumer_id.startswith("allreduce-ag") for edge in plan.edges)


def test_direct_plan_records_direct_request_metadata():
    inputs = resolved_input(CollectiveKind.ALL_REDUCE)

    plan = build_direct_plan(inputs, topology_for(inputs))

    assert plan.planning_mode is PlanningMode.DIRECT
    assert plan.planning_reason == "direct_request"


def test_alltoall_creates_one_source_destination_demand_per_slice():
    inputs = resolved_input(CollectiveKind.ALL_TO_ALL)

    plan = build_direct_plan(inputs, topology_for(inputs))

    assert len(plan.nodes) == 4
    assert all(
        node.local_collective.kind is CollectiveKind.ALL_TO_ALL
        for node in plan.nodes
    )
    assert all(len(node.logical_output.values) == 1 for node in plan.nodes)


@pytest.mark.parametrize("kind_name", ["scatter", "gather"])
def test_scatter_and_gather_are_not_direct_targets(kind_name):
    kind = CollectiveKind(kind_name)
    inputs = resolved_input(kind)

    with pytest.raises(InputValidationError, match="internal"):
        build_direct_plan(inputs, topology_for(inputs))


def test_scatter_and_gather_are_available_as_internal_nodes():
    inputs = resolved_input(CollectiveKind.ALL_GATHER)
    topology = topology_for(inputs)
    scattered = StageInterface(
        {
            OutputSlot(0, 0): frozenset({0}),
            OutputSlot(1, 0): frozenset({1}),
        }
    )
    gathered = StageInterface(
        {
            OutputSlot(0, 0): frozenset({0}),
            OutputSlot(0, 1): frozenset({1}),
        }
    )

    scatter_nodes = build_internal_scatter(0, (0, 1), scattered, topology)
    gather_nodes = build_internal_gather(0, (0, 1), gathered, topology)

    assert scatter_nodes[-1].logical_output == scattered
    assert gather_nodes[-1].logical_output == gathered
    assert scatter_nodes[-1].local_collective.kind is CollectiveKind.SCATTER
    assert gather_nodes[-1].local_collective.kind is CollectiveKind.GATHER


def single_value_interface():
    return StageInterface({OutputSlot(0, 0): frozenset({0})})


def simple_node(node_id):
    interface = single_value_interface()
    return PlanNode(
        node_id=node_id,
        stage_id=0,
        local_collective=CollectiveSpec(
            kind=CollectiveKind.BROADCAST,
            datatype="float32",
            root=0,
        ),
        communication_group=(0,),
        logical_input=interface,
        logical_output=interface,
        allowed_links=frozenset(),
        shared_resource_ids=frozenset(),
    )


def simple_plan(nodes, edges=()):
    interface = single_value_interface()
    return PlanDAG(
        collective=CollectiveSpec(
            kind=CollectiveKind.BROADCAST,
            datatype="float32",
            root=0,
        ),
        rank_count=1,
        slice_count=1,
        initial_inputs=interface,
        nodes=nodes,
        edges=edges,
        final_outputs=interface,
    )


def test_plan_rejects_duplicate_node_ids():
    with pytest.raises(SemanticError, match="unique"):
        simple_plan((simple_node("node"), simple_node("node")))


def test_plan_rejects_cycles_without_using_stage_barriers():
    interface = single_value_interface()
    nodes = (simple_node("a"), simple_node("b"))
    edges = (
        PlanEdge("a", "b", interface),
        PlanEdge("b", "a", interface),
    )

    with pytest.raises(SemanticError, match="acyclic"):
        simple_plan(nodes, edges)


def test_plan_rejects_edge_interface_not_produced_by_source():
    nodes = (simple_node("a"), simple_node("b"))
    mismatch = StageInterface({OutputSlot(0, 1): frozenset({0})})

    with pytest.raises(SemanticError, match="producer"):
        simple_plan(nodes, (PlanEdge("a", "b", mismatch),))


def test_plan_rejects_incorrect_final_interface():
    inputs = resolved_input(CollectiveKind.ALL_GATHER)
    plan = build_direct_plan(inputs, topology_for(inputs))
    incorrect = StageInterface(
        {OutputSlot(0, 0): frozenset({0})}
    )

    with pytest.raises(SemanticError, match="final outputs"):
        replace(plan, final_outputs=incorrect)


@pytest.mark.parametrize(
    "field,value",
    [
        ("planning_mode", "direct"),
        ("planning_reason", ""),
    ],
)
def test_plan_rejects_invalid_planning_metadata(field, value):
    with pytest.raises(SemanticError):
        replace(simple_plan((simple_node("node"),)), **{field: value})


def test_direct_plan_rejects_topology_rank_mismatch():
    inputs = resolved_input(CollectiveKind.ALL_GATHER)
    mismatched = replace(inputs, rank_count=3)

    with pytest.raises(InputValidationError, match="rank count"):
        build_direct_plan(mismatched, topology_for(inputs))


def test_plan_rejects_incorrect_initial_interface():
    inputs = resolved_input(CollectiveKind.ALL_GATHER)
    plan = build_direct_plan(inputs, topology_for(inputs))
    incomplete = StageInterface(
        {OutputSlot(0, 0): frozenset({0})}
    )

    with pytest.raises(SemanticError, match="initial inputs"):
        replace(plan, initial_inputs=incomplete)


def test_direct_plan_revalidates_partition_geometry():
    inputs = resolved_input(CollectiveKind.REDUCE_SCATTER, slice_count=3)

    with pytest.raises(InputValidationError, match="divisible"):
        build_direct_plan(inputs, topology_for(inputs))


def test_hierarchical_gateway_allreduce_excludes_direct_global_candidate():
    inputs = resolve_inputs(
        EXAMPLES / "topo" / "two_node_gateway.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    inputs = replace(
        inputs,
        strategies=replace(inputs.strategies, hierarchy=True),
    )

    with pytest.raises(InputValidationError, match="gateway plan"):
        build_direct_plan(inputs, load_topology(inputs))


def test_stage_interfaces_are_immutable_and_normalized():
    interface = StageInterface(
        {
            OutputSlot(1, 0): {1},
            OutputSlot(0, 0): {0},
        }
    )

    assert [value.slot for value in interface.logical_values] == [
        OutputSlot(0, 0),
        OutputSlot(1, 0),
    ]
    assert isinstance(interface.logical_values[0], LogicalValue)
    with pytest.raises(TypeError):
        interface.values[OutputSlot(0, 1)] = frozenset({1})


@pytest.mark.parametrize(
    "raw",
    [None, {}, {OutputSlot(0, 0): frozenset()}, {"slot": {0}}],
)
def test_stage_interface_rejects_invalid_values(raw):
    with pytest.raises(SemanticError):
        StageInterface(raw)


def test_plan_node_rejects_links_outside_its_group():
    interface = single_value_interface()

    with pytest.raises(SemanticError, match="leaves"):
        PlanNode(
            node_id="bad-link",
            stage_id=0,
            local_collective=CollectiveSpec(
                kind=CollectiveKind.BROADCAST,
                datatype="float32",
                root=0,
            ),
            communication_group=(0,),
            logical_input=interface,
            logical_output=interface,
            allowed_links=frozenset({LinkKey(0, 1)}),
            shared_resource_ids=frozenset(),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("node_id", ""),
        ("stage_id", -1),
        ("local_collective", object()),
        ("communication_group", None),
        ("communication_group", ()),
        ("communication_group", (1, 0)),
        ("logical_input", object()),
        ("logical_output", object()),
        ("allowed_links", None),
        ("allowed_links", frozenset({"link"})),
        ("shared_resource_ids", None),
        ("shared_resource_ids", frozenset({""})),
        ("dual_of_node_id", "node"),
    ],
)
def test_plan_node_rejects_invalid_fields(field, value):
    with pytest.raises(SemanticError):
        replace(simple_node("node"), **{field: value})


@pytest.mark.parametrize(
    "producer,consumer,interface",
    [
        ("", "b", None),
        ("a", "", None),
        ("a", "b", object()),
    ],
)
def test_plan_edge_rejects_invalid_fields(producer, consumer, interface):
    if interface is None:
        interface = single_value_interface()
    with pytest.raises(SemanticError):
        PlanEdge(producer, consumer, interface)


def test_plan_rejects_unknown_edge_nodes():
    interface = single_value_interface()

    with pytest.raises(SemanticError, match="unknown node"):
        simple_plan(
            (simple_node("a"),),
            (PlanEdge("a", "missing", interface),),
        )


def test_plan_rejects_consumer_interface_mismatch():
    interface = single_value_interface()
    consumer = replace(
        simple_node("b"),
        logical_input=StageInterface(
            {OutputSlot(0, 1): frozenset({0})}
        ),
    )

    with pytest.raises(SemanticError, match="consumer"):
        simple_plan(
            (simple_node("a"), consumer),
            (PlanEdge("a", "b", interface),),
        )


def test_plan_rejects_unavailable_node_input():
    unavailable = StageInterface(
        {OutputSlot(0, 1): frozenset({0})}
    )
    node = replace(simple_node("node"), logical_input=unavailable)

    with pytest.raises(SemanticError, match="not available"):
        simple_plan((node,))


def test_plan_rejects_duplicate_incoming_value_suppliers():
    interface = single_value_interface()
    nodes = (
        simple_node("a"),
        simple_node("b"),
        simple_node("consumer"),
    )
    edges = (
        PlanEdge("a", "consumer", interface),
        PlanEdge("b", "consumer", interface),
    )

    with pytest.raises(SemanticError, match="more than once"):
        simple_plan(nodes, edges)


def test_plan_rejects_unused_intermediate_output():
    unused = StageInterface(
        {
            OutputSlot(0, 0): frozenset({0}),
            OutputSlot(0, 1): frozenset({0}),
        }
    )
    node = replace(simple_node("node"), logical_output=unused)

    with pytest.raises(SemanticError, match="unused"):
        simple_plan((node,))


def test_plan_rejects_missing_final_producer():
    output = StageInterface(
        {OutputSlot(0, 1): frozenset({0})}
    )
    node = replace(simple_node("node"), logical_output=output)

    with pytest.raises(SemanticError, match="not produced"):
        simple_plan((node,))


@pytest.mark.parametrize(
    "node",
    [
        replace(simple_node("node"), communication_group=(1,)),
        replace(
            simple_node("node"),
            logical_output=StageInterface(
                {OutputSlot(0, 0): frozenset({1})}
            ),
        ),
    ],
)
def test_plan_rejects_out_of_range_node_interfaces(node):
    with pytest.raises(SemanticError, match="global range"):
        simple_plan((node,))


@pytest.mark.parametrize(
    "root,group,values,message",
    [
        (2, (0, 1), "scatter", "root"),
        (0, (1, 0), "scatter", "group"),
        (0, (0, 2), "scatter", "outside"),
        (0, (0, 1), "gather", "root"),
    ],
)
def test_internal_builders_reject_invalid_boundaries(
    root,
    group,
    values,
    message,
):
    inputs = resolved_input(CollectiveKind.ALL_GATHER)
    topology = topology_for(inputs)
    if values == "scatter":
        interface = StageInterface(
            {OutputSlot(0, 0): frozenset({0})}
        )
        builder = build_internal_scatter
    else:
        interface = StageInterface(
            {OutputSlot(1, 0): frozenset({0})}
        )
        builder = build_internal_gather

    with pytest.raises(InputValidationError, match=message):
        builder(root, group, interface, topology)
