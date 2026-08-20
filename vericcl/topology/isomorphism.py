from itertools import permutations
from typing import Tuple

from vericcl.errors import SemanticError
from vericcl.input.json_codec import sha256_json
from vericcl.topology.model import PerformanceCurve, Topology


# Exact enumeration is factorial. Seven external nodes require at most 5,040
# labelings; larger graphs retain raw identity and safely sacrifice reuse.
_EXACT_EXTERNAL_NODE_LIMIT = 7
_CANONICAL_LABELING = "canonical_permutation"
_IDENTITY_LABELING = "identity_fingerprint"


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


def _resource_member(
    topology: Topology,
    key: object,
    domain_set: set,
    relative_rank: dict,
    external_node_labels: dict,
    external_rank_positions: dict,
    labeling_mode: str,
) -> dict:
    def endpoint(rank: int) -> tuple:
        if rank in domain_set:
            return ("domain", relative_rank[rank])
        node = topology.node_membership[rank]
        if labeling_mode == _IDENTITY_LABELING:
            return (
                "external_identity",
                node,
                rank,
                external_rank_positions[rank],
                rank in topology.gateways,
            )
        return (
            "external",
            external_node_labels[node],
            external_rank_positions[rank],
            rank in topology.gateways,
        )

    edge = topology.links[key]
    return {
        "src": endpoint(key.src_rank),
        "dst": endpoint(key.dst_rank),
        "max_channels": edge.max_channels,
        "performance": _performance(edge.performance),
    }


def _external_rank_positions(topology: Topology) -> dict:
    node_ranks = {}
    for rank, node in topology.node_membership.items():
        node_ranks.setdefault(node, []).append(rank)
    return {
        rank: position
        for ranks in node_ranks.values()
        for position, rank in enumerate(sorted(ranks))
    }


def _external_nodes(
    topology: Topology,
    resource_ids: tuple,
    domain_set: set,
) -> tuple:
    nodes = set()
    for resource_id in resource_ids:
        resource = topology.shared_resources[resource_id]
        for key in resource.member_links:
            if key.src_rank not in domain_set:
                nodes.add(topology.node_membership[key.src_rank])
            if key.dst_rank not in domain_set:
                nodes.add(topology.node_membership[key.dst_rank])
    return tuple(sorted(nodes))


def _signature_value(
    topology: Topology,
    domain: tuple,
    domain_links: tuple,
    roles: list,
    relative_rank: dict,
    domain_set: set,
    external_node_labels: dict,
    external_rank_positions: dict,
    labeling_mode: str,
) -> dict:
    links = []
    for key, edge in domain_links:
        resources = []
        for resource_id in edge.resource_ids:
            resource = topology.shared_resources[resource_id]
            members = [
                _resource_member(
                    topology,
                    member,
                    domain_set,
                    relative_rank,
                    external_node_labels,
                    external_rank_positions,
                    labeling_mode,
                )
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
    return {
        "rank_count": len(domain),
        "roles": roles,
        "external_node_labeling": labeling_mode,
        "links": sorted(links, key=lambda item: (item["src"], item["dst"])),
    }


def exact_domain_signature(topology: Topology, ranks: tuple) -> str:
    if not isinstance(topology, Topology):
        raise SemanticError("topology must be a Topology")
    domain = _domain(topology, ranks)
    relative_rank = {rank: index for index, rank in enumerate(domain)}
    domain_set = set(domain)
    relative_node = {}
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
    domain_links = tuple(
        (key, edge)
        for key, edge in topology.links.items()
        if key.src_rank in domain_set and key.dst_rank in domain_set
    )
    resource_ids = tuple(
        sorted(
            {
                resource_id
                for _, edge in domain_links
                for resource_id in edge.resource_ids
            }
        )
    )
    external_nodes = _external_nodes(topology, resource_ids, domain_set)
    external_rank_positions = _external_rank_positions(topology)

    if len(external_nodes) > _EXACT_EXTERNAL_NODE_LIMIT:
        value = _signature_value(
            topology,
            domain,
            domain_links,
            roles,
            relative_rank,
            domain_set,
            {},
            external_rank_positions,
            _IDENTITY_LABELING,
        )
        return sha256_json(value)

    signatures = []
    for labels in permutations(range(len(external_nodes))):
        value = _signature_value(
            topology,
            domain,
            domain_links,
            roles,
            relative_rank,
            domain_set,
            dict(zip(external_nodes, labels)),
            external_rank_positions,
            _CANONICAL_LABELING,
        )
        signatures.append(sha256_json(value))
    return min(signatures)
