from types import MappingProxyType

import pytest

from vericcl.errors import SemanticError
from vericcl.topology.model import (
    DirectedLink,
    LinkKey,
    PerformanceCurve,
    SharedResource,
    Topology,
)


pytestmark = pytest.mark.phase02


def curve(alpha_us=1.0, invbw_us=2.0):
    return PerformanceCurve(alpha_us, invbw_us, {})


def link(src, dst, max_channels=4, resource_ids=()):
    return DirectedLink(
        key=LinkKey(src, dst),
        max_channels=max_channels,
        performance=curve(),
        resource_ids=resource_ids,
    )


def topology_with_links(*links, shared_resources=()):
    return Topology(
        rank_count=2,
        links={edge.key: edge for edge in links},
        shared_resources={item.resource_id: item for item in shared_resources},
        node_membership={0: 0, 1: 0},
        gateways=frozenset(),
        warnings=(),
    )


def test_links_are_directional():
    topology = topology_with_links(link(0, 1))

    assert topology.has_link(0, 1)
    assert not topology.has_link(1, 0)
    assert topology.destinations(0) == (1,)
    assert topology.sources(1) == (0,)


def test_lanes_are_directional_and_deterministic():
    edge = link(0, 1, max_channels=4)
    topology = topology_with_links(edge)

    lanes = topology.lanes(edge.key, channel_count=3)

    assert [(lane.src_rank, lane.dst_rank, lane.channel) for lane in lanes] == [
        (0, 1, 0),
        (0, 1, 1),
        (0, 1, 2),
    ]


def test_lane_count_must_not_exceed_link_limit():
    edge = link(0, 1, max_channels=2)
    topology = topology_with_links(edge)

    with pytest.raises(SemanticError, match="max_channels"):
        topology.lanes(edge.key, channel_count=3)


def test_resources_are_direction_specific():
    forward_key = LinkKey(0, 1)
    reverse_key = LinkKey(1, 0)
    forward_resource = SharedResource(
        resource_id="forward",
        member_links=(forward_key,),
        max_channels=4,
        performance=curve(),
    )
    reverse_resource = SharedResource(
        resource_id="reverse",
        member_links=(reverse_key,),
        max_channels=4,
        performance=curve(),
    )
    topology = topology_with_links(
        link(0, 1, resource_ids=("forward",)),
        link(1, 0, resource_ids=("reverse",)),
        shared_resources=(forward_resource, reverse_resource),
    )

    assert topology.resources_for(forward_key) == ("forward",)
    assert topology.resources_for(reverse_key) == ("reverse",)


def test_topology_mappings_are_immutable():
    topology = topology_with_links(link(0, 1))

    assert isinstance(topology.links, MappingProxyType)
    assert isinstance(topology.node_membership, MappingProxyType)
    with pytest.raises(TypeError):
        topology.links[LinkKey(1, 0)] = link(1, 0)


def test_topology_rejects_unknown_link_resource():
    edge = link(0, 1, resource_ids=("missing",))

    with pytest.raises(SemanticError, match="unknown shared resource"):
        topology_with_links(edge)
