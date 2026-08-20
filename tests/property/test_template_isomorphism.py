from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

from vericcl.input.loader import resolve_inputs
from vericcl.planner.model import PlanningMode, PlanNode, StageInterface
from vericcl.semantics.collective import (
    CollectiveKind,
    CollectiveSpec,
    OutputSlot,
)
from vericcl.solver.demands import SolverProblem, TransferDemand
from vericcl.solver.templates import build_solver_templates
from vericcl.topology.model import (
    DirectedLink,
    LinkKey,
    PerformanceCurve,
    SharedResource,
    Topology,
)


pytestmark = pytest.mark.phase03


EXAMPLES = Path(__file__).parents[2] / "vericcl" / "examples"


def _inputs(slice_count=8, slice_size=1):
    base = resolve_inputs(
        EXAMPLES / "topo" / "two_rank.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    return replace(
        base,
        rank_count=4,
        collective=CollectiveSpec(
            kind=CollectiveKind.ALL_GATHER,
            datatype="float32",
        ),
        hyperparameters=replace(
            base.hyperparameters,
            total_size_bytes=slice_count * slice_size,
            slice_size_bytes=slice_size,
        ),
        strategies=replace(base.strategies, symmetry=False),
    )


def _topology(
    *,
    second_channels=4,
    second_invbw=2.0,
    second_resource_channels=2,
    second_resource_invbw=2.0,
    second_has_resource=True,
    omit_second_reverse=False,
):
    groups = ((0, 1), (2, 3))
    links = {}
    resources = {}
    for group_index, group in enumerate(groups):
        keys = tuple(
            LinkKey(src, dst)
            for src in group
            for dst in group
            if src != dst
            and not (
                group_index == 1
                and omit_second_reverse
                and (src, dst) == (3, 2)
            )
        )
        has_resource = group_index == 0 or second_has_resource
        resource_id = "rail-resource-{}".format(group_index)
        for key in keys:
            links[key] = DirectedLink(
                key=key,
                max_channels=4 if group_index == 0 else second_channels,
                performance=PerformanceCurve(
                    alpha_us=1.0,
                    invbw_us=2.0 if group_index == 0 else second_invbw,
                    bandwidth_bytes_per_us={1: 1024.0},
                ),
                resource_ids=(resource_id,) if has_resource else (),
            )
        if has_resource:
            resources[resource_id] = SharedResource(
                resource_id=resource_id,
                member_links=keys,
                max_channels=2
                if group_index == 0
                else second_resource_channels,
                performance=PerformanceCurve(
                    alpha_us=1.0,
                    invbw_us=2.0
                    if group_index == 0
                    else second_resource_invbw,
                    bandwidth_bytes_per_us={1: 512.0},
                ),
            )
    return Topology(
        rank_count=4,
        links=links,
        shared_resources=resources,
        node_membership={0: 0, 1: 0, 2: 1, 3: 1},
        gateways=frozenset({0, 2}),
        warnings=(),
    )


def _topology_with_external_resource(external_dst):
    resource_id = "shared-rail"
    member_links = (
        LinkKey(0, 1),
        LinkKey(1, 0),
        LinkKey(0, external_dst),
    )
    links = {
        key: DirectedLink(
            key=key,
            max_channels=4,
            performance=PerformanceCurve(
                alpha_us=1.0,
                invbw_us=2.0,
                bandwidth_bytes_per_us={1: 1024.0},
            ),
            resource_ids=(resource_id,),
        )
        for key in member_links
    }
    return Topology(
        rank_count=4,
        links=links,
        shared_resources={
            resource_id: SharedResource(
                resource_id=resource_id,
                member_links=member_links,
                max_channels=2,
                performance=PerformanceCurve(
                    alpha_us=1.0,
                    invbw_us=2.0,
                    bandwidth_bytes_per_us={1: 512.0},
                ),
            )
        },
        node_membership={0: 0, 1: 0, 2: 1, 3: 1},
        gateways=frozenset({0}),
        warnings=(),
    )


def _resource_graph_topology(
    group,
    external_rank,
    *,
    add_external_coresource=False,
    signature="caller-supplied-resource-graph",
):
    domain_key = LinkKey(group[0], group[1])
    external_key = LinkKey(group[0], external_rank)
    primary_id = "primary-{}".format(group[0])
    coresource_id = "co-{}".format(group[0])
    external_resources = (
        (primary_id, coresource_id)
        if add_external_coresource
        else (primary_id,)
    )
    curve = PerformanceCurve(
        alpha_us=1.0,
        invbw_us=2.0,
        bandwidth_bytes_per_us={1: 1024.0},
    )
    links = {
        domain_key: DirectedLink(
            key=domain_key,
            max_channels=4,
            performance=curve,
            resource_ids=(primary_id,),
        ),
        external_key: DirectedLink(
            key=external_key,
            max_channels=4,
            performance=curve,
            resource_ids=external_resources,
        ),
    }
    resources = {
        primary_id: SharedResource(
            resource_id=primary_id,
            member_links=(domain_key, external_key),
            max_channels=2,
            performance=curve,
        )
    }
    if add_external_coresource:
        resources[coresource_id] = SharedResource(
            resource_id=coresource_id,
            member_links=(external_key,),
            max_channels=1,
            performance=curve,
        )
    rank_count = 6
    membership = {
        rank: rank for rank in range(rank_count)
    }
    membership[group[1]] = membership[group[0]]
    return Topology(
        rank_count=rank_count,
        links=links,
        shared_resources=resources,
        node_membership=membership,
        gateways=frozenset({group[0]}),
        warnings=(),
        isomorphism_signature=signature,
    )


def _problem(inputs, topology, group, root, logical_position, node_id):
    contributor = root * inputs.hyperparameters.slice_count + logical_position
    contributors = frozenset({contributor})
    allowed = frozenset(
        key
        for key in topology.links
        if key.src_rank in group and key.dst_rank in group
    )
    leaf = next(rank for rank in group if rank != root)
    demand = TransferDemand(
        demand_id="{}-demand".format(node_id),
        node_id=node_id,
        stage_id=0,
        root_rank=root,
        required_leaf_rank=leaf,
        logical_position=logical_position,
        contributors=contributors,
        member_slice_ids=contributors,
        allowed_links=allowed,
        legal_links=allowed,
        forbidden_members=(),
        candidate_paths=((root, leaf),),
    )
    node = PlanNode(
        node_id=node_id,
        stage_id=0,
        local_collective=CollectiveSpec(
            kind=CollectiveKind.BROADCAST,
            datatype="float32",
            root=root,
        ),
        communication_group=tuple(group),
        logical_input=StageInterface(
            {OutputSlot(root, logical_position): contributors}
        ),
        logical_output=StageInterface(
            {
                OutputSlot(rank, logical_position): contributors
                for rank in group
            }
        ),
        allowed_links=allowed,
        shared_resource_ids=frozenset(
            resource_id
            for key in allowed
            for resource_id in topology.link(key).resource_ids
        ),
    )
    return SolverProblem(
        node=node,
        inputs=inputs,
        topology=topology,
        demands=(demand,),
        candidate_edges=frozenset(),
        infeasible_demand_ids=(),
        restrictions=(),
    )


@given(
    representative_position=st.integers(min_value=0, max_value=7),
    member_position=st.integers(min_value=0, max_value=7),
)
def test_exact_rank_renumbering_has_invertible_member_maps(
    representative_position,
    member_position,
):
    inputs = _inputs()
    topology = _topology()
    first = _problem(
        inputs,
        topology,
        (0, 1),
        0,
        representative_position,
        "group-a",
    )
    second = _problem(
        inputs,
        topology,
        (2, 3),
        2,
        member_position,
        "group-b",
    )

    templates = build_solver_templates(
        (first, second),
        PlanningMode.GATEWAY_ALLGATHER,
    )

    assert len(templates) == 1
    member = next(item for item in templates[0].members if item.node_id == "group-b")
    assert member.rank_map == ((0, 2), (1, 3))
    assert dict(member.logical_position_map) == {
        representative_position: member_position
    }
    assert len(set(dict(member.contributor_map).values())) == len(
        member.contributor_map
    )


@pytest.mark.parametrize(
    "topology_arguments",
    (
        {"omit_second_reverse": True},
        {"second_channels": 3},
        {"second_invbw": 2.5},
        {"second_has_resource": False},
        {"second_resource_channels": 1},
        {"second_resource_invbw": 2.5},
    ),
)
def test_any_resource_or_link_change_splits_exact_classes(topology_arguments):
    inputs = _inputs()
    topology = _topology(**topology_arguments)
    first = _problem(inputs, topology, (0, 1), 0, 0, "group-a")
    second = _problem(inputs, topology, (2, 3), 2, 1, "group-b")

    templates = build_solver_templates(
        (first, second),
        PlanningMode.GATEWAY_ALLGATHER,
    )

    assert len(templates) == 2


def test_external_shared_resource_endpoints_cannot_merge_unsafely():
    inputs = _inputs()
    first_topology = _topology_with_external_resource(2)
    second_topology = _topology_with_external_resource(3)
    first = _problem(inputs, first_topology, (0, 1), 0, 0, "external-a")
    second = _problem(inputs, second_topology, (0, 1), 0, 1, "external-b")

    templates = build_solver_templates(
        (first, second),
        PlanningMode.GATEWAY_ALLGATHER,
    )

    assert len(templates) == 2


def test_member_link_resource_comembership_splits_exact_templates():
    inputs = replace(_inputs(), rank_count=6)
    baseline_topology = _resource_graph_topology((0, 1), 4)
    changed_topology = _resource_graph_topology(
        (0, 1),
        4,
        add_external_coresource=True,
    )
    baseline = _problem(
        inputs,
        baseline_topology,
        (0, 1),
        0,
        0,
        "comembership-a",
    )
    changed = _problem(
        inputs,
        changed_topology,
        (0, 1),
        0,
        1,
        "comembership-b",
    )

    templates = build_solver_templates(
        (baseline, changed),
        PlanningMode.GATEWAY_ALLGATHER,
    )

    assert len(templates) == 2


def test_external_resource_graph_rank_renumbering_reuses_exact_template():
    inputs = replace(_inputs(), rank_count=6)
    first_topology = _resource_graph_topology((0, 1), 4)
    second_topology = _resource_graph_topology((2, 3), 5)
    first = _problem(
        inputs,
        first_topology,
        (0, 1),
        0,
        0,
        "renumbered-a",
    )
    second = _problem(
        inputs,
        second_topology,
        (2, 3),
        2,
        1,
        "renumbered-b",
    )

    templates = build_solver_templates(
        (first, second),
        PlanningMode.GATEWAY_ALLGATHER,
    )

    assert len(templates) == 1
    translated = next(
        member
        for member in templates[0].members
        if member.node_id == "renumbered-b"
    )
    assert translated.rank_map == ((0, 2), (1, 3))
