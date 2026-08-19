from dataclasses import dataclass
from typing import Dict, Set, Tuple

from vericcl.errors import SemanticError
from vericcl.topology.model import Topology


def _groups(value: object, field: str) -> Tuple[Tuple[int, ...], ...]:
    try:
        groups = tuple(tuple(group) for group in value)
    except TypeError as error:
        raise SemanticError("{} must contain rank groups".format(field)) from error
    for group in groups:
        if not group or group != tuple(sorted(group)):
            raise SemanticError("{} groups must be non-empty and sorted".format(field))
        if len(group) != len(set(group)):
            raise SemanticError("{} groups must contain unique ranks".format(field))
        if any(
            isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
            for rank in group
        ):
            raise SemanticError("{} groups contain invalid ranks".format(field))
    if groups != tuple(sorted(groups)) or len(groups) != len(set(groups)):
        raise SemanticError("{} must be sorted and unique".format(field))
    return groups


@dataclass(frozen=True)
class CommunicationGroups:
    intra_node: Tuple[Tuple[int, ...], ...]
    inter_node: Tuple[Tuple[int, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "intra_node",
            _groups(self.intra_node, "intra_node"),
        )
        inter_node = _groups(self.inter_node, "inter_node")
        if any(len(group) < 2 for group in inter_node):
            raise SemanticError("inter_node groups must contain at least two ranks")
        object.__setattr__(self, "inter_node", inter_node)


def _components(
    topology: Topology,
    candidates: Tuple[int, ...],
) -> Tuple[Tuple[int, ...], ...]:
    adjacency: Dict[int, Set[int]] = {rank: set() for rank in candidates}
    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            if topology.has_link(left, right) and topology.has_link(right, left):
                adjacency[left].add(right)
                adjacency[right].add(left)
    components = []
    remaining = set(candidates)
    while remaining:
        root = min(remaining)
        pending = [root]
        component = set()
        while pending:
            rank = pending.pop()
            if rank in component:
                continue
            component.add(rank)
            pending.extend(sorted(adjacency[rank] - component, reverse=True))
        remaining.difference_update(component)
        if len(component) >= 2:
            components.append(tuple(sorted(component)))
    return tuple(components)


def discover_communication_groups(topology: Topology) -> CommunicationGroups:
    if not isinstance(topology, Topology):
        raise SemanticError("topology must be a Topology")
    by_node = {}
    for rank, node in topology.node_membership.items():
        by_node.setdefault(node, []).append(rank)
    intra_node = tuple(
        sorted(tuple(sorted(ranks)) for ranks in by_node.values())
    )
    gateways_by_node = [
        tuple(rank for rank in ranks if rank in topology.gateways)
        for ranks in intra_node
    ]
    position_count = max(
        (len(gateways) for gateways in gateways_by_node),
        default=0,
    )
    inter_node = []
    for position in range(position_count):
        candidates = tuple(
            gateways[position]
            for gateways in gateways_by_node
            if position < len(gateways)
        )
        inter_node.extend(_components(topology, candidates))
    return CommunicationGroups(
        intra_node=intra_node,
        inter_node=tuple(sorted(inter_node)),
    )


def gateway_rank_correspondence(
    topology: Topology,
    groups: CommunicationGroups,
) -> Tuple[Tuple[int, ...], ...]:
    if not isinstance(topology, Topology):
        raise SemanticError("topology must be a Topology")
    if not isinstance(groups, CommunicationGroups):
        raise SemanticError("groups must be CommunicationGroups")
    if len(groups.intra_node) < 2:
        return ()
    covered_ranks = set()
    covered_nodes = set()
    gateways_by_node = []
    for group in groups.intra_node:
        nodes = {topology.node_membership[rank] for rank in group}
        if len(nodes) != 1 or covered_ranks.intersection(group):
            return ()
        node = next(iter(nodes))
        if node in covered_nodes:
            return ()
        covered_ranks.update(group)
        covered_nodes.add(node)
        gateways_by_node.append(
            tuple(rank for rank in group if rank in topology.gateways)
        )
    if covered_ranks != set(range(topology.rank_count)):
        return ()
    gateway_counts = {len(gateways) for gateways in gateways_by_node}
    if len(gateway_counts) != 1 or not next(iter(gateway_counts)):
        return ()
    rail_count = len(gateways_by_node[0])
    return tuple(
        tuple(gateways[rail_index] for gateways in gateways_by_node)
        for rail_index in range(rail_count)
    )
