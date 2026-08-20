from itertools import permutations
from math import factorial
from typing import Tuple

from vericcl.errors import SemanticError
from vericcl.input.json_codec import sha256_json
from vericcl.topology.model import PerformanceCurve, Topology


# Exact enumeration is factorial in both external nodes and shared resources.
# At most 5,040 joint labelings are evaluated. Larger graphs retain raw
# identity and safely sacrifice reuse rather than risk a false-positive match.
_MAX_CANONICAL_LABELINGS = 5040
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
    resource_labels: dict,
    labeling_mode: str,
    pinned_external_rank_labels: dict,
) -> dict:
    def endpoint(rank: int) -> tuple:
        if rank in domain_set:
            return ("domain", relative_rank[rank])
        node = topology.node_membership[rank]
        if labeling_mode == _IDENTITY_LABELING:
            identity = (
                "external_identity",
                node,
                rank,
                external_rank_positions[rank],
                rank in topology.gateways,
            )
            if rank in pinned_external_rank_labels:
                return identity + (
                    "pinned",
                    pinned_external_rank_labels[rank],
                )
            return identity
        if rank in pinned_external_rank_labels:
            return (
                "external_pinned",
                pinned_external_rank_labels[rank],
                external_node_labels[node],
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
        "resources": tuple(
            sorted(resource_labels[item] for item in edge.resource_ids)
        ),
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


def _relevant_resources(
    topology: Topology,
    domain_links: tuple,
    selected_resource_ids: tuple,
) -> tuple:
    pending = list(selected_resource_ids)
    pending.extend(
        resource_id
        for _, edge in domain_links
        for resource_id in edge.resource_ids
    )
    relevant = set()
    while pending:
        resource_id = pending.pop()
        if resource_id in relevant:
            continue
        relevant.add(resource_id)
        resource = topology.shared_resources[resource_id]
        for key in resource.member_links:
            pending.extend(topology.links[key].resource_ids)
    return tuple(sorted(relevant))


def _signature_value(
    topology: Topology,
    domain: tuple,
    domain_links: tuple,
    roles: list,
    relative_rank: dict,
    domain_set: set,
    external_node_labels: dict,
    external_rank_positions: dict,
    resource_ids: tuple,
    resource_labels: dict,
    selected_resource_ids: frozenset,
    labeling_mode: str,
    pinned_external_rank_labels: dict,
) -> dict:
    links = []
    for key, edge in domain_links:
        links.append(
            {
                "src": relative_rank[key.src_rank],
                "dst": relative_rank[key.dst_rank],
                "max_channels": edge.max_channels,
                "performance": _performance(edge.performance),
                "resources": tuple(
                    sorted(
                        resource_labels[resource_id]
                        for resource_id in edge.resource_ids
                    )
                ),
            }
        )
    resources = []
    for resource_id in resource_ids:
        resource = topology.shared_resources[resource_id]
        members = [
            _resource_member(
                topology,
                member,
                domain_set,
                relative_rank,
                external_node_labels,
                external_rank_positions,
                resource_labels,
                labeling_mode,
                pinned_external_rank_labels,
            )
            for member in resource.member_links
        ]
        resources.append(
            {
                "label": resource_labels[resource_id],
                "selected": resource_id in selected_resource_ids,
                "members": sorted(
                    members,
                    key=lambda value: sha256_json(value),
                ),
                "max_channels": resource.max_channels,
                "performance": _performance(resource.performance),
            }
        )
    return {
        "rank_count": len(domain),
        "roles": roles,
        "external_node_labeling": labeling_mode,
        "links": sorted(links, key=lambda item: (item["src"], item["dst"])),
        "resources": sorted(resources, key=lambda item: item["label"]),
    }


def exact_domain_signature(
    topology: Topology,
    ranks: tuple,
    selected_resource_ids: object = (),
    pinned_external_rank_labels: object = None,
) -> str:
    if not isinstance(topology, Topology):
        raise SemanticError("topology must be a Topology")
    domain = _domain(topology, ranks)
    if pinned_external_rank_labels is None:
        pinned_external_rank_labels = {}
    try:
        pinned_external_rank_labels = dict(pinned_external_rank_labels)
    except (TypeError, ValueError) as error:
        raise SemanticError(
            "pinned external rank labels must be a mapping"
        ) from error
    for rank, label in pinned_external_rank_labels.items():
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise SemanticError("pinned external ranks must be integers")
        if rank < 0 or rank >= topology.rank_count:
            raise SemanticError("pinned external rank is outside the topology")
        if rank in domain:
            raise SemanticError("pinned external rank belongs to the domain")
        if isinstance(label, bool) or not isinstance(label, int) or label < 0:
            raise SemanticError(
                "pinned external rank labels must be non-negative integers"
            )
    if len(set(pinned_external_rank_labels.values())) != len(
        pinned_external_rank_labels
    ):
        raise SemanticError("pinned external rank labels must be unique")
    try:
        selected_resource_ids = tuple(sorted(set(selected_resource_ids)))
    except TypeError as error:
        raise SemanticError("selected resource IDs must be iterable") from error
    if any(
        not isinstance(resource_id, str) or not resource_id
        for resource_id in selected_resource_ids
    ):
        raise SemanticError("selected resource IDs must be non-empty strings")
    if not set(selected_resource_ids) <= set(topology.shared_resources):
        raise SemanticError("selected resource ID is unknown")
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
    resource_ids = _relevant_resources(
        topology,
        domain_links,
        selected_resource_ids,
    )
    external_nodes = _external_nodes(topology, resource_ids, domain_set)
    external_rank_positions = _external_rank_positions(topology)

    labeling_count = factorial(len(external_nodes)) * factorial(
        len(resource_ids)
    )
    if labeling_count > _MAX_CANONICAL_LABELINGS:
        value = _signature_value(
            topology,
            domain,
            domain_links,
            roles,
            relative_rank,
            domain_set,
            {},
            external_rank_positions,
            resource_ids,
            {
                resource_id: ("resource_identity", resource_id)
                for resource_id in resource_ids
            },
            frozenset(selected_resource_ids),
            _IDENTITY_LABELING,
            pinned_external_rank_labels,
        )
        return sha256_json(value)

    signatures = []
    for external_labels in permutations(range(len(external_nodes))):
        for resource_label_values in permutations(range(len(resource_ids))):
            value = _signature_value(
                topology,
                domain,
                domain_links,
                roles,
                relative_rank,
                domain_set,
                dict(zip(external_nodes, external_labels)),
                external_rank_positions,
                resource_ids,
                dict(zip(resource_ids, resource_label_values)),
                frozenset(selected_resource_ids),
                _CANONICAL_LABELING,
                pinned_external_rank_labels,
            )
            signatures.append(sha256_json(value))
    return min(signatures)


def exact_domain_mapping_is_valid(
    source_topology: Topology,
    source_ranks: tuple,
    source_selected_resource_ids: object,
    target_topology: Topology,
    target_ranks: tuple,
    target_selected_resource_ids: object,
    rank_mapping: object,
) -> bool:
    if not isinstance(source_topology, Topology) or not isinstance(
        target_topology,
        Topology,
    ):
        raise SemanticError("mapping topologies must be Topology values")
    source_domain = _domain(source_topology, source_ranks)
    target_domain = _domain(target_topology, target_ranks)
    if len(source_domain) != len(target_domain):
        return False
    try:
        pairs = tuple(tuple(pair) for pair in rank_mapping)
    except TypeError as error:
        raise SemanticError("rank mapping must be an iterable of pairs") from error
    if any(len(pair) != 2 for pair in pairs):
        raise SemanticError("rank mapping must contain pairs")
    for source, target in pairs:
        if (
            isinstance(source, bool)
            or not isinstance(source, int)
            or source < 0
            or source >= source_topology.rank_count
            or isinstance(target, bool)
            or not isinstance(target, int)
            or target < 0
            or target >= target_topology.rank_count
        ):
            raise SemanticError("rank mapping contains an invalid rank")
    sources = tuple(source for source, _ in pairs)
    targets = tuple(target for _, target in pairs)
    if len(sources) != len(set(sources)) or len(targets) != len(set(targets)):
        return False
    mapping = dict(pairs)
    if tuple(mapping.get(rank) for rank in source_domain) != target_domain:
        return False
    source_domain_set = set(source_domain)
    target_domain_set = set(target_domain)
    external_pairs = tuple(
        (source, target)
        for source, target in sorted(pairs)
        if source not in source_domain_set or target not in target_domain_set
    )
    if any(
        (source in source_domain_set) != (target in target_domain_set)
        for source, target in external_pairs
    ):
        return False
    source_pins = {
        source: label
        for label, (source, _) in enumerate(external_pairs)
    }
    target_pins = {
        target: label
        for label, (_, target) in enumerate(external_pairs)
    }
    return exact_domain_signature(
        source_topology,
        source_domain,
        source_selected_resource_ids,
        source_pins,
    ) == exact_domain_signature(
        target_topology,
        target_domain,
        target_selected_resource_ids,
        target_pins,
    )
