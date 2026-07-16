from pathlib import Path

import pytest

from vericcl.input.loader import resolve_inputs
from vericcl.topology.loader import load_topology, topology_from_mapping
from vericcl.topology.model import LinkKey


pytestmark = pytest.mark.phase02


EXAMPLES = Path(__file__).parents[3] / "vericcl" / "examples"


def load_example(name):
    resolved = resolve_inputs(
        EXAMPLES / "topo" / name,
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    return load_topology(resolved)


def test_channels_share_directed_link_resources():
    topology = load_example("two_node_gateway.json")

    resources = topology.resources_for(LinkKey(0, 4))

    assert "inter-node-0-to-1" in resources
    assert "nic-node-0-egress" in resources
    assert "nic-node-1-ingress" in resources


def test_reverse_link_uses_independent_directional_resources():
    topology = load_example("two_node_gateway.json")

    forward = set(topology.resources_for(LinkKey(0, 4)))
    reverse = set(topology.resources_for(LinkKey(4, 0)))

    assert forward.isdisjoint(reverse)


def test_shared_resource_membership_is_direction_specific():
    topology = load_example("two_node_gateway.json")
    resource = topology.shared_resources["nic-node-0-egress"]

    assert resource.member_links == (LinkKey(0, 4),)
    assert LinkKey(4, 0) not in resource.member_links


def test_inconsistent_link_parameters_emit_deterministic_warning():
    raw = {
        "ranks": 2,
        "nodes": [{"id": 0, "ranks": [0, 1], "gateways": []}],
        "directed_links": [
            {
                "src": 0,
                "dst": 1,
                "max_channels": 2,
                "alpha": 2,
                "beta": 4,
                "invbw": 5,
                "resources": [],
            }
        ],
        "shared_resources": [],
    }

    topology = topology_from_mapping(raw)

    assert topology.warnings == (
        "link 0->1: performance parameters disagree; invbw is authoritative",
    )
