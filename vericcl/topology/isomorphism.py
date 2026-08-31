from typing import Tuple

from vericcl.errors import SemanticError
from vericcl.input.json_codec import sha256_json
from vericcl.topology.model import PerformanceCurve, Topology


def _domain(topology: Topology, ranks: object) -> Tuple[int, ...]:
    if not isinstance(ranks, tuple) or not ranks:
        raise SemanticError("domain ranks must be a non-empty tuple")
    for rank in ranks:
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise SemanticError("domain ranks must be integers")
        if rank < 0 or rank >= topology.rank_count:
            raise SemanticError("domain rank is outside the topology")
    if ranks != tuple(sorted(ranks)) or len(ranks) != len(set(ranks)):
        raise SemanticError("domain ranks must be sorted and unique")
    return ranks


def _performance(curve: PerformanceCurve) -> dict:
    return {
        "alpha_us": curve.alpha_us,
        "invbw_us": curve.invbw_us,
        "bandwidth_bytes_per_us": [
            (concurrency, bandwidth)
            for concurrency, bandwidth in curve.bandwidth_bytes_per_us.items()
        ],
    }


def exact_domain_signature(topology: Topology, ranks: tuple) -> str:
    if not isinstance(topology, Topology):
        raise SemanticError("topology must be a Topology")
    domain = _domain(topology, ranks)
    relative_rank = {rank: index for index, rank in enumerate(domain)}
    domain_set = set(domain)
    relative_node = {}
    node_members = {}
    for rank, node in topology.node_membership.items():
        node_members.setdefault(node, []).append(rank)
    roles = []
    for rank in domain:
        node = topology.node_membership[rank]
        if node not in relative_node:
            relative_node[node] = len(relative_node)
        roles.append(
            {
                "node": relative_node[node],
                "gateway": rank in topology.gateways,
            }
        )

    links = []
    for key, edge in topology.links.items():
        if key.src_rank not in domain_set or key.dst_rank not in domain_set:
            continue
        resources = []
        for resource_id in edge.resource_ids:
            resource = topology.shared_resources[resource_id]
            members = [
                {
                    "src": _relative_endpoint(
                        topology,
                        member.src_rank,
                        relative_rank,
                        relative_node,
                        node_members,
                    ),
                    "dst": _relative_endpoint(
                        topology,
                        member.dst_rank,
                        relative_rank,
                        relative_node,
                        node_members,
                    ),
                    "same_node": (
                        topology.node_membership[member.src_rank]
                        == topology.node_membership[member.dst_rank]
                    ),
                }
                for member in resource.member_links
            ]
            resources.append(
                {
                    "members": sorted(
                        members,
                        key=lambda value: sha256_json(value),
                    ),
                    "max_channels": resource.max_channels,
                    "performance": _performance(resource.performance),
                }
            )
        links.append(
            {
                "src": relative_rank[key.src_rank],
                "dst": relative_rank[key.dst_rank],
                "max_channels": edge.max_channels,
                "performance": _performance(edge.performance),
                "resources": sorted(
                    resources,
                    key=lambda value: sha256_json(value),
                ),
            }
        )
    value = {
        "rank_count": len(domain),
        "roles": roles,
        "links": sorted(links, key=lambda item: (item["src"], item["dst"])),
    }
    return sha256_json(value)


def _relative_endpoint(
    topology: Topology,
    rank: int,
    relative_rank: dict,
    relative_node: dict,
    node_members: dict,
) -> dict:
    if rank in relative_rank:
        return {
            "scope": "domain",
            "rank": relative_rank[rank],
        }
    node = topology.node_membership[rank]
    members = node_members[node]
    return {
        "scope": "external",
        "domain_node": relative_node.get(node),
        "node_size": len(members),
        "node_position": members.index(rank),
        "gateway": rank in topology.gateways,
    }
