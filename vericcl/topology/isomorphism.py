from typing import Tuple

from vericcl.errors import SemanticError
from vericcl.input.json_codec import canonical_json, sha256_json
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


# Bound the exact individualization/refinement search. Exceeding the bound is
# an explicit semantic error; the planner never accepts an approximate match.
_MAX_CANONICAL_STATES = 100000


def _refine_colors(colors: tuple, edges: tuple) -> tuple:
    outgoing = [[] for _ in colors]
    incoming = [[] for _ in colors]
    for src, dst, label in edges:
        outgoing[src].append((label, dst))
        incoming[dst].append((label, src))
    refined = colors
    while True:
        signatures = [
            (
                refined[vertex],
                tuple(
                    sorted(
                        (label, refined[dst])
                        for label, dst in outgoing[vertex]
                    )
                ),
                tuple(
                    sorted(
                        (label, refined[src])
                        for label, src in incoming[vertex]
                    )
                ),
            )
            for vertex in range(len(refined))
        ]
        palette = {
            signature: color
            for color, signature in enumerate(sorted(set(signatures)))
        }
        updated = tuple(palette[signature] for signature in signatures)
        if updated == refined:
            return refined
        refined = updated


def _canonical_colored_graph(vertex_colors: tuple, edges: tuple) -> str:
    """Return the exact canonical hash of a bounded colored graph search."""
    palette = {
        color: index for index, color in enumerate(sorted(set(vertex_colors)))
    }
    initial = tuple(palette[color] for color in vertex_colors)
    state_count = [0]

    def search(colors: tuple) -> str:
        state_count[0] += 1
        if state_count[0] > _MAX_CANONICAL_STATES:
            raise SemanticError(
                "domain isomorphism canonicalization limit exceeded"
            )
        refined = _refine_colors(colors, edges)
        classes = {}
        for vertex, color in enumerate(refined):
            classes.setdefault(color, []).append(vertex)
        ambiguous = [
            (len(vertices), color, tuple(vertices))
            for color, vertices in classes.items()
            if len(vertices) > 1
        ]
        if not ambiguous:
            order = tuple(
                vertex
                for vertex, _ in sorted(
                    enumerate(refined),
                    key=lambda item: item[1],
                )
            )
            relative = {
                vertex: index for index, vertex in enumerate(order)
            }
            return canonical_json(
                {
                    "colors": [vertex_colors[vertex] for vertex in order],
                    "edges": sorted(
                        (
                            relative[src],
                            relative[dst],
                            label,
                        )
                        for src, dst, label in edges
                    ),
                }
            )
        _, _, vertices = min(ambiguous)
        candidates = []
        marker = max(refined) + 1
        for vertex in vertices:
            individualized = list(refined)
            individualized[vertex] = marker
            candidates.append(search(tuple(individualized)))
        return min(candidates)

    return sha256_json(search(initial))


def _resource_incidence_signature(
    topology: Topology,
    domain_set: set,
    relative_rank: dict,
    relative_node: dict,
    node_members: dict,
) -> str:
    vertex_colors = []
    edges = []

    def vertex(color: dict) -> int:
        vertex_id = len(vertex_colors)
        vertex_colors.append(canonical_json(color))
        return vertex_id

    rank_vertices = {}
    node_vertices = {}

    def rank_vertex(rank: int) -> int:
        if rank in rank_vertices:
            return rank_vertices[rank]
        if rank in relative_rank:
            color = {
                "kind": "domain_rank",
                "rank": relative_rank[rank],
            }
        else:
            node = topology.node_membership[rank]
            members = node_members[node]
            color = {
                "kind": "external_rank",
                "domain_node": relative_node.get(node),
                "node_size": len(members),
                "gateway": rank in topology.gateways,
            }
        rank_vertices[rank] = vertex(color)
        if rank not in relative_rank:
            node = topology.node_membership[rank]
            if node not in node_vertices:
                node_vertices[node] = vertex(
                    {
                        "kind": "external_node",
                        "domain_node": relative_node.get(node),
                        "node_size": len(node_members[node]),
                    }
                )
            edges.append(
                (node_vertices[node], rank_vertices[rank], "contains")
            )
        return rank_vertices[rank]

    link_vertices = {}
    referenced_resources = set()
    for key, edge in topology.links.items():
        if key.src_rank not in domain_set or key.dst_rank not in domain_set:
            continue
        link_vertices[key] = vertex(
            {
                "kind": "domain_link",
                "src": relative_rank[key.src_rank],
                "dst": relative_rank[key.dst_rank],
            }
        )
        for resource_id in edge.resource_ids:
            referenced_resources.add(resource_id)

    resource_vertices = {}
    for resource_id in referenced_resources:
        resource = topology.shared_resources[resource_id]
        resource_vertices[resource_id] = vertex(
            {
                "kind": "resource",
                "max_channels": resource.max_channels,
                "performance": _performance(resource.performance),
            }
        )
    for key, edge in topology.links.items():
        if key not in link_vertices:
            continue
        for resource_id in edge.resource_ids:
            edges.append(
                (
                    link_vertices[key],
                    resource_vertices[resource_id],
                    "uses",
                )
            )

    for resource_id in referenced_resources:
        resource = topology.shared_resources[resource_id]
        for member in resource.member_links:
            member_vertex = vertex({"kind": "resource_member"})
            edges.append(
                (
                    resource_vertices[resource_id],
                    member_vertex,
                    "member",
                )
            )
            edges.append(
                (member_vertex, rank_vertex(member.src_rank), "src")
            )
            edges.append(
                (member_vertex, rank_vertex(member.dst_rank), "dst")
            )

    return _canonical_colored_graph(tuple(vertex_colors), tuple(edges))


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
    resource_incidence = _resource_incidence_signature(
        topology,
        domain_set,
        relative_rank,
        relative_node,
        node_members,
    )

    links = []
    for key, edge in topology.links.items():
        if key.src_rank not in domain_set or key.dst_rank not in domain_set:
            continue
        links.append(
            {
                "src": relative_rank[key.src_rank],
                "dst": relative_rank[key.dst_rank],
                "max_channels": edge.max_channels,
                "performance": _performance(edge.performance),
            }
        )
    value = {
        "rank_count": len(domain),
        "roles": roles,
        "links": sorted(links, key=lambda item: (item["src"], item["dst"])),
        "resource_incidence": resource_incidence,
    }
    return sha256_json(value)
