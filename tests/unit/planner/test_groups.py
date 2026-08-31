from dataclasses import replace
from pathlib import Path

import pytest

from vericcl.errors import SemanticError
from vericcl.input.loader import resolve_inputs
from vericcl.planner.build import build_plan
from vericcl.planner.groups import (
    CommunicationGroups,
    discover_communication_groups,
    eligible_gateway_groups,
)
from vericcl.planner.model import PlanningMode
from vericcl.semantics.collective import CollectiveKind, CollectiveSpec
from vericcl.topology.isomorphism import exact_domain_signature
from vericcl.topology.loader import load_topology, topology_from_mapping


pytestmark = pytest.mark.phase02


EXAMPLES = Path(__file__).parents[3] / "vericcl" / "examples"


def gateway_topology():
    resolved = resolve_inputs(
        EXAMPLES / "topo" / "two_node_gateway.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    return load_topology(resolved)


def domain_topology(
    reverse=True,
    max_channels=4,
    resource_members=None,
    invbw=2,
    gateways=(0,),
):
    links = [
        {
            "src": 0,
            "dst": 1,
            "alpha": 1,
            "invbw": invbw,
            "max_channels": max_channels,
            "resources": ["fabric"] if resource_members else [],
        }
    ]
    if reverse:
        links.append(
            {
                "src": 1,
                "dst": 0,
                "alpha": 1,
                "invbw": invbw,
                "max_channels": max_channels,
                "resources": [],
            }
        )
    resources = []
    if resource_members:
        resources.append(
            {
                "id": "fabric",
                "member_links": resource_members,
                "alpha": 1,
                "invbw": 2,
                "max_channels": max_channels,
            }
        )
    return topology_from_mapping(
        {
            "ranks": 2,
            "nodes": [
                {"id": 7, "ranks": [0, 1], "gateways": list(gateways)}
            ],
            "directed_links": links,
            "shared_resources": resources,
        }
    )


def external_alias_topology():
    return topology_from_mapping(
        {
            "ranks": 6,
            "nodes": [
                {"id": 17, "ranks": [0, 1], "gateways": [0]},
                {"id": 31, "ranks": [2, 3], "gateways": [2]},
                {"id": 53, "ranks": [4, 5], "gateways": [4]},
            ],
            "directed_links": [
                {
                    "src": 0,
                    "dst": 1,
                    "alpha": 1,
                    "invbw": 2,
                    "resources": ["fabric-a"],
                },
                {"src": 1, "dst": 0, "alpha": 1, "invbw": 2},
                {
                    "src": 2,
                    "dst": 3,
                    "alpha": 1,
                    "invbw": 2,
                    "resources": ["fabric-b"],
                },
                {"src": 3, "dst": 2, "alpha": 1, "invbw": 2},
                {
                    "src": 4,
                    "dst": 5,
                    "alpha": 1,
                    "invbw": 2,
                    "resources": ["fabric-c"],
                },
                {"src": 5, "dst": 4, "alpha": 1, "invbw": 2},
                {
                    "src": 0,
                    "dst": 2,
                    "alpha": 2,
                    "invbw": 3,
                    "resources": ["fabric-a"],
                },
                {
                    "src": 1,
                    "dst": 2,
                    "alpha": 2,
                    "invbw": 3,
                    "resources": ["fabric-a"],
                },
                {
                    "src": 2,
                    "dst": 0,
                    "alpha": 2,
                    "invbw": 3,
                    "resources": ["fabric-b"],
                },
                {
                    "src": 3,
                    "dst": 4,
                    "alpha": 2,
                    "invbw": 3,
                    "resources": ["fabric-b"],
                },
                {
                    "src": 4,
                    "dst": 0,
                    "alpha": 2,
                    "invbw": 3,
                    "resources": ["fabric-c"],
                },
                {
                    "src": 5,
                    "dst": 0,
                    "alpha": 2,
                    "invbw": 3,
                    "resources": ["fabric-c"],
                },
                {"src": 2, "dst": 4, "alpha": 2, "invbw": 3},
                {"src": 4, "dst": 2, "alpha": 2, "invbw": 3},
            ],
            "shared_resources": [
                {
                    "id": "fabric-a",
                    "member_links": [[0, 1], [0, 2], [1, 2]],
                    "alpha": 1,
                    "invbw": 2,
                    "max_channels": 8,
                },
                {
                    "id": "fabric-b",
                    "member_links": [[2, 3], [2, 0], [3, 4]],
                    "alpha": 1,
                    "invbw": 2,
                    "max_channels": 8,
                },
                {
                    "id": "fabric-c",
                    "member_links": [[4, 5], [4, 0], [5, 0]],
                    "alpha": 1,
                    "invbw": 2,
                    "max_channels": 8,
                },
            ],
        }
    )


def test_gateway_groups_do_not_invent_rank_pairs():
    groups = discover_communication_groups(gateway_topology())

    assert groups.intra_node == ((0, 1, 2, 3), (4, 5, 6, 7))
    assert groups.inter_node == ((0, 4),)
    assert (1, 5) not in groups.inter_node


def test_eligible_gateway_groups_select_every_real_positional_rail():
    raw = {
        "ranks": 8,
        "nodes": [
            {"id": 0, "ranks": [0, 1, 2, 3], "gateways": [0, 1, 2, 3]},
            {"id": 1, "ranks": [4, 5, 6, 7], "gateways": [4, 5, 6, 7]},
        ],
        "directed_links": [
            {"src": src, "dst": dst, "alpha": 1, "invbw": 2}
            for base in (0, 4)
            for src in range(base, base + 4)
            for dst in range(base, base + 4)
            if src != dst
        ]
        + [
            {"src": src, "dst": dst, "alpha": 2, "invbw": 3}
            for rail in range(4)
            for src, dst in ((rail, rail + 4), (rail + 4, rail))
        ],
        "shared_resources": [],
    }
    topology = topology_from_mapping(raw)
    groups = discover_communication_groups(topology)

    assert eligible_gateway_groups(topology, groups) == (
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )


def test_eligible_gateway_groups_reject_unequal_gateway_counts():
    raw = {
        "ranks": 4,
        "nodes": [
            {"id": 0, "ranks": [0, 1], "gateways": [0]},
            {"id": 1, "ranks": [2, 3], "gateways": [2, 3]},
        ],
        "directed_links": [
            {"src": 0, "dst": 1, "alpha": 1, "invbw": 2},
            {"src": 1, "dst": 0, "alpha": 1, "invbw": 2},
            {"src": 2, "dst": 3, "alpha": 1, "invbw": 2},
            {"src": 3, "dst": 2, "alpha": 1, "invbw": 2},
            {"src": 0, "dst": 2, "alpha": 2, "invbw": 3},
            {"src": 2, "dst": 0, "alpha": 2, "invbw": 3},
        ],
        "shared_resources": [],
    }
    topology = topology_from_mapping(raw)

    assert eligible_gateway_groups(
        topology,
        discover_communication_groups(topology),
    ) == ()


def test_eligible_gateway_groups_reject_nonisomorphic_local_domains():
    raw = {
        "ranks": 4,
        "nodes": [
            {"id": 0, "ranks": [0, 1], "gateways": [0]},
            {"id": 1, "ranks": [2, 3], "gateways": [2]},
        ],
        "directed_links": [
            {"src": 0, "dst": 1, "alpha": 1, "invbw": 2},
            {"src": 1, "dst": 0, "alpha": 1, "invbw": 2},
            {"src": 2, "dst": 3, "alpha": 1, "invbw": 3},
            {"src": 3, "dst": 2, "alpha": 1, "invbw": 2},
            {"src": 0, "dst": 2, "alpha": 2, "invbw": 3},
            {"src": 2, "dst": 0, "alpha": 2, "invbw": 3},
        ],
        "shared_resources": [],
    }
    topology = topology_from_mapping(raw)

    assert eligible_gateway_groups(
        topology,
        discover_communication_groups(topology),
    ) == ()


def test_eligible_gateway_groups_recheck_bidirectional_connectivity():
    raw = {
        "ranks": 4,
        "nodes": [
            {"id": 0, "ranks": [0, 1], "gateways": [0]},
            {"id": 1, "ranks": [2, 3], "gateways": [2]},
        ],
        "directed_links": [
            {"src": 0, "dst": 1, "alpha": 1, "invbw": 2},
            {"src": 1, "dst": 0, "alpha": 1, "invbw": 2},
            {"src": 2, "dst": 3, "alpha": 1, "invbw": 2},
            {"src": 3, "dst": 2, "alpha": 1, "invbw": 2},
            {"src": 0, "dst": 2, "alpha": 2, "invbw": 3},
        ],
        "shared_resources": [],
    }
    topology = topology_from_mapping(raw)
    forged = CommunicationGroups(
        intra_node=((0, 1), (2, 3)),
        inter_node=((0, 2),),
    )

    assert eligible_gateway_groups(topology, forged) == ()


def test_declared_gateway_without_cross_link_is_not_an_inter_node_group():
    raw = {
        "ranks": 4,
        "nodes": [
            {"id": 0, "ranks": [0, 1], "gateways": [0]},
            {"id": 1, "ranks": [2, 3], "gateways": [2]},
        ],
        "directed_links": [
            {"src": 0, "dst": 1, "alpha": 1, "invbw": 2},
            {"src": 1, "dst": 0, "alpha": 1, "invbw": 2},
            {"src": 2, "dst": 3, "alpha": 1, "invbw": 2},
            {"src": 3, "dst": 2, "alpha": 1, "invbw": 2},
        ],
        "shared_resources": [],
    }

    groups = discover_communication_groups(topology_from_mapping(raw))

    assert groups.inter_node == ()


def test_gateway_positions_form_independent_cross_node_groups():
    raw = {
        "ranks": 4,
        "nodes": [
            {"id": 0, "ranks": [0, 1], "gateways": [0, 1]},
            {"id": 1, "ranks": [2, 3], "gateways": [2, 3]},
        ],
        "directed_links": [
            {"src": 0, "dst": 2, "alpha": 1, "invbw": 2},
            {"src": 2, "dst": 0, "alpha": 1, "invbw": 2},
            {"src": 1, "dst": 3, "alpha": 1, "invbw": 2},
            {"src": 3, "dst": 1, "alpha": 1, "invbw": 2},
        ],
        "shared_resources": [],
    }

    groups = discover_communication_groups(topology_from_mapping(raw))

    assert groups.inter_node == ((0, 2), (1, 3))


def test_partial_gateway_connectivity_keeps_only_real_components():
    raw = {
        "ranks": 3,
        "nodes": [
            {"id": 0, "ranks": [0], "gateways": [0]},
            {"id": 1, "ranks": [1], "gateways": [1]},
            {"id": 2, "ranks": [2], "gateways": [2]},
        ],
        "directed_links": [
            {"src": 0, "dst": 1, "alpha": 1, "invbw": 2},
            {"src": 1, "dst": 0, "alpha": 1, "invbw": 2},
        ],
        "shared_resources": [],
    }

    groups = discover_communication_groups(topology_from_mapping(raw))

    assert groups.inter_node == ((0, 1),)


def test_isomorphic_node_domains_have_equal_relative_signatures():
    topology = gateway_topology()

    assert exact_domain_signature(topology, (0, 1, 2, 3)) == (
        exact_domain_signature(topology, (4, 5, 6, 7))
    )


def test_signature_changes_with_direction_capacity_performance_and_roles():
    baseline = exact_domain_signature(domain_topology(), (0, 1))
    directed = exact_domain_signature(domain_topology(reverse=False), (0, 1))
    capacity = exact_domain_signature(
        domain_topology(max_channels=2),
        (0, 1),
    )
    shared = exact_domain_signature(
        domain_topology(resource_members=[[0, 1]]),
        (0, 1),
    )
    performance = exact_domain_signature(domain_topology(invbw=3), (0, 1))
    roles = exact_domain_signature(domain_topology(gateways=()), (0, 1))

    assert len({baseline, directed, capacity, shared, performance, roles}) == 6


def test_signature_includes_cross_domain_resource_membership():
    def topology_with_external_member(include_external_member):
        members = [[0, 1]]
        if include_external_member:
            members.append([0, 2])
        return topology_from_mapping(
            {
                "ranks": 3,
                "nodes": [
                    {"id": 0, "ranks": [0, 1], "gateways": [0]},
                    {"id": 1, "ranks": [2], "gateways": [2]},
                ],
                "directed_links": [
                    {
                        "src": 0,
                        "dst": 1,
                        "alpha": 1,
                        "invbw": 2,
                        "resources": ["fabric"],
                    },
                    {
                        "src": 0,
                        "dst": 2,
                        "alpha": 2,
                        "invbw": 3,
                        "resources": (
                            ["fabric"] if include_external_member else []
                        ),
                    },
                ],
                "shared_resources": [
                    {
                        "id": "fabric",
                        "member_links": members,
                        "alpha": 1,
                        "invbw": 2,
                    }
                ],
            }
        )

    local_only = exact_domain_signature(
        topology_with_external_member(False),
        (0, 1),
    )
    cross_domain = exact_domain_signature(
        topology_with_external_member(True),
        (0, 1),
    )

    assert local_only != cross_domain


def test_signature_preserves_external_node_aliasing_after_renumbering():
    topology = external_alias_topology()
    same_external_node = exact_domain_signature(topology, (0, 1))
    different_external_nodes = exact_domain_signature(topology, (2, 3))
    renumbered_same_external_node = exact_domain_signature(topology, (4, 5))

    assert same_external_node == renumbered_same_external_node
    assert same_external_node != different_external_nodes
    assert eligible_gateway_groups(
        topology,
        discover_communication_groups(topology),
    ) == ()


def test_external_node_alias_mismatch_falls_back_to_direct_allgather():
    inputs = resolve_inputs(
        EXAMPLES / "topo" / "two_node_gateway.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    inputs = replace(
        inputs,
        rank_count=6,
        collective=CollectiveSpec(
            kind=CollectiveKind.ALL_GATHER,
            datatype="float32",
        ),
        strategies=replace(inputs.strategies, hierarchy=True),
    )

    plan = build_plan(inputs, external_alias_topology())

    assert plan.planning_mode is PlanningMode.DIRECT
    assert plan.planning_reason == "no_eligible_gateway_domain"


def test_signature_is_deterministic():
    topology = gateway_topology()

    assert exact_domain_signature(topology, (0, 4)) == exact_domain_signature(
        topology,
        (0, 4),
    )


@pytest.mark.parametrize("ranks", [(1, 0), (0, 0), (), (0, 2)])
def test_signature_rejects_invalid_domains(ranks):
    with pytest.raises(SemanticError):
        exact_domain_signature(domain_topology(), ranks)


def test_group_discovery_requires_topology_model():
    with pytest.raises(SemanticError, match="Topology"):
        discover_communication_groups({})


@pytest.mark.parametrize(
    "intra,inter",
    [
        (None, ()),
        (((1, 0),), ()),
        (((0, 0),), ()),
        (((True,),), ()),
        (((1,), (0,)), ()),
        (((0,),), ((0,),)),
    ],
)
def test_communication_groups_reject_invalid_records(intra, inter):
    with pytest.raises(SemanticError):
        CommunicationGroups(intra_node=intra, inter_node=inter)


def test_three_node_gateway_component_is_deterministic():
    raw = {
        "ranks": 3,
        "nodes": [
            {"id": rank, "ranks": [rank], "gateways": [rank]}
            for rank in range(3)
        ],
        "directed_links": [
            {"src": src, "dst": dst, "alpha": 1, "invbw": 2}
            for src in range(3)
            for dst in range(3)
            if src != dst
        ],
        "shared_resources": [],
    }

    groups = discover_communication_groups(topology_from_mapping(raw))

    assert groups.inter_node == ((0, 1, 2),)
