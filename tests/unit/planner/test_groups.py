from pathlib import Path

import pytest

from vericcl.errors import SemanticError
from vericcl.input.loader import resolve_inputs
from vericcl.planner.groups import (
    CommunicationGroups,
    discover_communication_groups,
)
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


def test_gateway_groups_do_not_invent_rank_pairs():
    groups = discover_communication_groups(gateway_topology())

    assert groups.intra_node == ((0, 1, 2, 3), (4, 5, 6, 7))
    assert groups.inter_node == ((0, 4),)
    assert (1, 5) not in groups.inter_node


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


def test_signature_includes_external_shared_resource_membership():
    raw = {
        "ranks": 4,
        "nodes": [
            {"id": 0, "ranks": [0, 1], "gateways": [0]},
            {"id": 1, "ranks": [2, 3], "gateways": [2]},
        ],
        "directed_links": [
            {
                "src": 0,
                "dst": 1,
                "alpha": 1,
                "invbw": 2,
                "resources": ["left-fabric"],
            },
            {"src": 1, "dst": 0, "alpha": 1, "invbw": 2},
            {
                "src": 1,
                "dst": 2,
                "alpha": 1,
                "invbw": 2,
                "resources": ["left-fabric"],
            },
            {
                "src": 2,
                "dst": 3,
                "alpha": 1,
                "invbw": 2,
                "resources": ["right-fabric"],
            },
            {"src": 3, "dst": 2, "alpha": 1, "invbw": 2},
        ],
        "shared_resources": [
            {
                "id": "left-fabric",
                "member_links": [[0, 1], [1, 2]],
                "alpha": 1,
                "invbw": 2,
            },
            {
                "id": "right-fabric",
                "member_links": [[2, 3]],
                "alpha": 1,
                "invbw": 2,
            },
        ],
    }
    topology = topology_from_mapping(raw)

    assert exact_domain_signature(topology, (0, 1)) != (
        exact_domain_signature(topology, (2, 3))
    )


def test_signature_preserves_external_node_coreference_without_raw_ids():
    raw = {
        "ranks": 10,
        "nodes": [
            {"id": 10, "ranks": [0, 1], "gateways": [0]},
            {"id": 20, "ranks": [2, 3], "gateways": [2]},
            {"id": 30, "ranks": [4, 5], "gateways": [4]},
            {"id": 40, "ranks": [6], "gateways": []},
            {"id": 50, "ranks": [7], "gateways": []},
            {"id": 60, "ranks": [8], "gateways": []},
            {"id": 70, "ranks": [9], "gateways": []},
        ],
        "directed_links": [
            {
                "src": 0,
                "dst": 1,
                "alpha": 1,
                "invbw": 2,
                "resources": ["converged-a"],
            },
            {"src": 1, "dst": 0, "alpha": 1, "invbw": 2},
            {
                "src": 0,
                "dst": 6,
                "alpha": 1,
                "invbw": 2,
                "resources": ["converged-a"],
            },
            {
                "src": 1,
                "dst": 6,
                "alpha": 1,
                "invbw": 2,
                "resources": ["converged-a"],
            },
            {
                "src": 2,
                "dst": 3,
                "alpha": 1,
                "invbw": 2,
                "resources": ["diverged"],
            },
            {"src": 3, "dst": 2, "alpha": 1, "invbw": 2},
            {
                "src": 2,
                "dst": 7,
                "alpha": 1,
                "invbw": 2,
                "resources": ["diverged"],
            },
            {
                "src": 3,
                "dst": 8,
                "alpha": 1,
                "invbw": 2,
                "resources": ["diverged"],
            },
            {
                "src": 4,
                "dst": 5,
                "alpha": 1,
                "invbw": 2,
                "resources": ["converged-b"],
            },
            {"src": 5, "dst": 4, "alpha": 1, "invbw": 2},
            {
                "src": 4,
                "dst": 9,
                "alpha": 1,
                "invbw": 2,
                "resources": ["converged-b"],
            },
            {
                "src": 5,
                "dst": 9,
                "alpha": 1,
                "invbw": 2,
                "resources": ["converged-b"],
            },
        ],
        "shared_resources": [
            {
                "id": "converged-a",
                "member_links": [[0, 1], [0, 6], [1, 6]],
                "alpha": 1,
                "invbw": 2,
            },
            {
                "id": "diverged",
                "member_links": [[2, 3], [2, 7], [3, 8]],
                "alpha": 1,
                "invbw": 2,
            },
            {
                "id": "converged-b",
                "member_links": [[4, 5], [4, 9], [5, 9]],
                "alpha": 1,
                "invbw": 2,
            },
        ],
    }
    topology = topology_from_mapping(raw)
    converged_a = exact_domain_signature(topology, (0, 1))
    diverged = exact_domain_signature(topology, (2, 3))
    converged_b = exact_domain_signature(topology, (4, 5))

    assert converged_a != diverged
    assert converged_a == converged_b


def external_cycle_resource_topology():
    resource_specs = (
        (
            "six-cycle",
            (0, 1),
            ((6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 6)),
        ),
        (
            "two-three-cycles",
            (2, 3),
            ((12, 13), (13, 14), (14, 12), (15, 16), (16, 17), (17, 15)),
        ),
        (
            "renumbered-six-cycle",
            (4, 5),
            ((18, 21), (21, 19), (19, 23), (23, 20), (20, 22), (22, 18)),
        ),
    )
    directed_links = []
    resources = []
    for resource_id, domain_link, external_links in resource_specs:
        member_links = (domain_link,) + external_links
        directed_links.extend(
            {
                "src": src,
                "dst": dst,
                "alpha": 1,
                "invbw": 2,
                "resources": [resource_id],
            }
            for src, dst in member_links
        )
        directed_links.append(
            {
                "src": domain_link[1],
                "dst": domain_link[0],
                "alpha": 1,
                "invbw": 2,
            }
        )
        resources.append(
            {
                "id": resource_id,
                "member_links": member_links,
                "alpha": 1,
                "invbw": 2,
            }
        )
    external_node_ids = (
        101,
        103,
        105,
        107,
        109,
        111,
        202,
        204,
        206,
        208,
        210,
        212,
        905,
        901,
        909,
        903,
        907,
        900,
    )
    return topology_from_mapping(
        {
            "ranks": 24,
            "nodes": [
                {"id": 10, "ranks": [0, 1], "gateways": [0]},
                {"id": 20, "ranks": [2, 3], "gateways": [2]},
                {"id": 30, "ranks": [4, 5], "gateways": [4]},
            ]
            + [
                {"id": node_id, "ranks": [rank], "gateways": []}
                for rank, node_id in zip(range(6, 24), external_node_ids)
            ],
            "directed_links": directed_links,
            "shared_resources": resources,
        }
    )


def test_signature_distinguishes_complete_external_resource_member_graphs():
    topology = external_cycle_resource_topology()

    assert exact_domain_signature(topology, (0, 1)) != (
        exact_domain_signature(topology, (2, 3))
    )


def test_signature_is_invariant_to_external_graph_renumbering():
    topology = external_cycle_resource_topology()

    assert exact_domain_signature(topology, (0, 1)) == (
        exact_domain_signature(topology, (4, 5))
    )


def test_signature_preserves_external_node_coreference_across_resources():
    resource_members = {
        "shared-forward": ((0, 1), (4, 5)),
        "shared-reverse": ((1, 0), (4, 6)),
        "distinct-forward": ((2, 3), (7, 9)),
        "distinct-reverse": ((3, 2), (8, 10)),
    }
    directed_links = [
        {
            "src": src,
            "dst": dst,
            "alpha": 1,
            "invbw": 2,
            "resources": [resource_id],
        }
        for resource_id, members in resource_members.items()
        for src, dst in members
    ]
    topology = topology_from_mapping(
        {
            "ranks": 11,
            "nodes": [
                {"id": 10, "ranks": [0, 1], "gateways": [0]},
                {"id": 20, "ranks": [2, 3], "gateways": [2]},
            ]
            + [
                {"id": 100 + rank, "ranks": [rank], "gateways": []}
                for rank in range(4, 11)
            ],
            "directed_links": directed_links,
            "shared_resources": [
                {
                    "id": resource_id,
                    "member_links": members,
                    "alpha": 1,
                    "invbw": 2,
                }
                for resource_id, members in resource_members.items()
            ],
        }
    )

    assert exact_domain_signature(topology, (0, 1)) != (
        exact_domain_signature(topology, (2, 3))
    )


def test_signature_fallback_does_not_merge_large_external_graphs():
    resource_specs = (
        (
            "nine-cycle",
            (0, 1),
            tuple((rank, 4 + (rank - 3) % 9) for rank in range(4, 13)),
        ),
        (
            "three-three-cycles",
            (2, 3),
            (
                (13, 14),
                (14, 15),
                (15, 13),
                (16, 17),
                (17, 18),
                (18, 16),
                (19, 20),
                (20, 21),
                (21, 19),
            ),
        ),
    )
    directed_links = []
    resources = []
    for resource_id, domain_link, external_links in resource_specs:
        members = (domain_link,) + external_links
        directed_links.extend(
            {
                "src": src,
                "dst": dst,
                "alpha": 1,
                "invbw": 2,
                "resources": [resource_id],
            }
            for src, dst in members
        )
        directed_links.append(
            {
                "src": domain_link[1],
                "dst": domain_link[0],
                "alpha": 1,
                "invbw": 2,
            }
        )
        resources.append(
            {
                "id": resource_id,
                "member_links": members,
                "alpha": 1,
                "invbw": 2,
            }
        )
    topology = topology_from_mapping(
        {
            "ranks": 22,
            "nodes": [
                {"id": 10, "ranks": [0, 1], "gateways": [0]},
                {"id": 20, "ranks": [2, 3], "gateways": [2]},
            ]
            + [
                {"id": 100 + rank, "ranks": [rank], "gateways": []}
                for rank in range(4, 22)
            ],
            "directed_links": directed_links,
            "shared_resources": resources,
        }
    )

    assert exact_domain_signature(topology, (0, 1)) != (
        exact_domain_signature(topology, (2, 3))
    )
