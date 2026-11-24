from dataclasses import dataclass
from typing import Dict, FrozenSet, Mapping, Optional, Tuple

from vericcl.errors import SemanticError
from vericcl.input.json_codec import sha256_json
from vericcl.input.models import ForbiddenTransfer
from vericcl.planner.model import PlanNode, PlanningMode, StageInterface
from vericcl.semantics.collective import CollectiveKind, CollectiveSpec
from vericcl.solver.demands import (
    SolverProblem,
    TransferDemand,
    routing_unit_key,
)
from vericcl.topology.model import LinkKey, PerformanceCurve, Topology


_CHAIN_KINDS = frozenset(
    {
        CollectiveKind.GATHER,
        CollectiveKind.SCATTER,
        CollectiveKind.ALL_TO_ALL,
    }
)


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SemanticError("{} must be a non-empty string".format(field))
    return value


def _mapping(value: object, field: str) -> Tuple[Tuple[int, int], ...]:
    try:
        entries = tuple(tuple(item) for item in value)
    except (TypeError, ValueError) as error:
        raise SemanticError("{} must contain pairs".format(field)) from error
    if any(
        len(item) != 2
        or isinstance(item[0], bool)
        or isinstance(item[1], bool)
        or not isinstance(item[0], int)
        or not isinstance(item[1], int)
        or item[0] < 0
        or item[1] < 0
        for item in entries
    ):
        raise SemanticError("{} must contain non-negative integer pairs".format(field))
    if entries != tuple(sorted(entries)):
        raise SemanticError("{} must be sorted".format(field))
    sources = tuple(item[0] for item in entries)
    targets = tuple(item[1] for item in entries)
    if len(sources) != len(set(sources)) or len(targets) != len(set(targets)):
        raise SemanticError("{} must be a bijection".format(field))
    return entries


@dataclass(frozen=True)
class RoutingUnit:
    unit_id: str
    node: PlanNode
    demands: Tuple[TransferDemand, ...]

    def __post_init__(self) -> None:
        _identifier(self.unit_id, "routing_unit.unit_id")
        if not isinstance(self.node, PlanNode):
            raise SemanticError("routing_unit.node must be a PlanNode")
        demands = tuple(self.demands)
        if not demands or not all(
            isinstance(demand, TransferDemand) for demand in demands
        ):
            raise SemanticError(
                "routing_unit.demands must contain TransferDemand values"
            )
        if any(
            demand.node_id != self.node.node_id
            or demand.stage_id != self.node.stage_id
            for demand in demands
        ):
            raise SemanticError("routing unit demand is outside its plan node")
        demand_ids = tuple(demand.demand_id for demand in demands)
        if len(demand_ids) != len(set(demand_ids)):
            raise SemanticError("routing unit demand IDs must be unique")
        object.__setattr__(
            self,
            "demands",
            tuple(sorted(demands, key=lambda item: item.demand_id)),
        )


@dataclass(frozen=True)
class TemplateMember:
    unit_id: str
    node_id: str
    rank_map: Tuple[Tuple[int, int], ...]
    contributor_map: Tuple[Tuple[int, int], ...]
    logical_position_map: Tuple[Tuple[int, int], ...]

    def __post_init__(self) -> None:
        _identifier(self.unit_id, "template_member.unit_id")
        _identifier(self.node_id, "template_member.node_id")
        object.__setattr__(
            self,
            "rank_map",
            _mapping(self.rank_map, "template_member.rank_map"),
        )
        object.__setattr__(
            self,
            "contributor_map",
            _mapping(
                self.contributor_map,
                "template_member.contributor_map",
            ),
        )
        object.__setattr__(
            self,
            "logical_position_map",
            _mapping(
                self.logical_position_map,
                "template_member.logical_position_map",
            ),
        )


@dataclass(frozen=True)
class SolverTemplate:
    template_id: str
    representative: RoutingUnit
    members: Tuple[TemplateMember, ...]
    exact_signature: str
    # True when the representative was solved for an earlier plan (template
    # library); then none of this plan's units is the representative itself.
    library_representative: bool = False

    def __post_init__(self) -> None:
        _identifier(self.template_id, "solver_template.template_id")
        _identifier(self.exact_signature, "solver_template.exact_signature")
        if not isinstance(self.representative, RoutingUnit):
            raise SemanticError(
                "solver_template.representative must be a RoutingUnit"
            )
        members = tuple(self.members)
        if not members or not all(
            isinstance(member, TemplateMember) for member in members
        ):
            raise SemanticError(
                "solver_template.members must contain TemplateMember values"
            )
        # Unit IDs are unique within a plan node; instantiation identifies a
        # member by (node, unit), so that pair must be unique in a template.
        unit_keys = tuple((member.node_id, member.unit_id) for member in members)
        if len(unit_keys) != len(set(unit_keys)):
            raise SemanticError("solver template unit IDs must be unique")
        unit_ids = {member.unit_id for member in members}
        if not self.library_representative and self.representative.unit_id not in unit_ids:
            raise SemanticError("solver template omits its representative")
        object.__setattr__(self, "members", members)


def split_routing_units(problem: SolverProblem) -> Tuple[RoutingUnit, ...]:
    if not isinstance(problem, SolverProblem):
        raise SemanticError("problem must be a SolverProblem")
    if not problem.demands:
        return ()
    grouped: Dict[object, list] = {}
    if problem.node.local_collective.kind in _CHAIN_KINDS:
        for demand in problem.demands:
            grouped[(demand.demand_id,)] = [demand]
    else:
        for demand in problem.demands:
            grouped.setdefault(routing_unit_key(demand), []).append(demand)
    units = tuple(
        RoutingUnit(
            unit_id="{}-u{:08d}".format(problem.node.node_id, index),
            node=problem.node,
            demands=tuple(demands),
        )
        for index, (_, demands) in enumerate(sorted(grouped.items()))
    )
    original_ids = tuple(sorted(demand.demand_id for demand in problem.demands))
    split_ids = tuple(
        sorted(demand.demand_id for unit in units for demand in unit.demands)
    )
    if split_ids != original_ids or len(split_ids) != len(set(split_ids)):
        raise SemanticError("routing unit split changed the demand set")
    return units


def _unit_positions(unit: RoutingUnit) -> Tuple[int, ...]:
    return tuple(sorted({demand.logical_position for demand in unit.demands}))


def _unit_contributors(unit: RoutingUnit) -> FrozenSet[int]:
    values = set()
    for demand in unit.demands:
        values.update(demand.contributors)
        values.update(demand.member_slice_ids)
        values.update(item.slice_id for item in demand.forbidden_members)
    return frozenset(values)


def _rank_map(
    representative: RoutingUnit,
    member: RoutingUnit,
) -> Optional[Dict[int, int]]:
    source = representative.node.communication_group
    target = member.node.communication_group
    if len(source) != len(target):
        return None
    if source == target:
        return {rank: rank for rank in source}
    return dict(zip(source, target))


def _position_map(
    representative: RoutingUnit,
    member: RoutingUnit,
) -> Optional[Dict[int, int]]:
    source = _unit_positions(representative)
    target = _unit_positions(member)
    if len(source) != len(target):
        return None
    return dict(zip(source, target))


def _contributor_map(
    representative: RoutingUnit,
    representative_problem: SolverProblem,
    member: RoutingUnit,
    member_problem: SolverProblem,
    rank_map: Mapping[int, int],
    position_map: Mapping[int, int],
) -> Optional[Dict[int, int]]:
    source = _unit_contributors(representative)
    target = _unit_contributors(member)
    if len(source) != len(target):
        return None
    mapped = {}
    source_slice_count = representative_problem.slice_count
    target_slice_count = member_problem.slice_count
    target_by_position: Dict[int, list] = {}
    for contributor in sorted(target):
        _, logical_position = divmod(contributor, target_slice_count)
        target_by_position.setdefault(logical_position, []).append(contributor)
    source_by_position: Dict[int, list] = {}
    for contributor in sorted(source):
        _, logical_position = divmod(contributor, source_slice_count)
        if logical_position not in position_map:
            return None
        source_by_position.setdefault(
            position_map[logical_position],
            [],
        ).append(contributor)
    if set(source_by_position) != set(target_by_position):
        return None
    for logical_position, source_values in sorted(source_by_position.items()):
        remaining = set(target_by_position[logical_position])
        if len(source_values) != len(remaining):
            return None
        for contributor in source_values:
            source_rank, _ = divmod(contributor, source_slice_count)
            preferred_ranks = []
            if source_rank in rank_map:
                preferred_ranks.append(rank_map[source_rank])
            if source_rank not in preferred_ranks:
                preferred_ranks.append(source_rank)
            candidates = tuple(
                rank * target_slice_count + logical_position
                for rank in preferred_ranks
            )
            candidate = next(
                (value for value in candidates if value in remaining),
                None,
            )
            if candidate is None:
                return None
            mapped[contributor] = candidate
            remaining.remove(candidate)
    return mapped


def _curve_payload(curve: PerformanceCurve) -> tuple:
    return (
        curve.alpha_us,
        curve.invbw_us,
        tuple(curve.bandwidth_bytes_per_us.items()),
        curve.beta_us_per_byte,
    )


_PAYLOAD_CACHE: Dict[tuple, tuple] = {}


def _cached_payload(kind, topology, scope, rank_map, compute):
    # Pure functions of (topology, group or node resources, rank map); units
    # and their automorphism candidates recompute them very often.
    key = (kind, id(topology), scope, tuple(sorted(rank_map.items())))
    cached = _PAYLOAD_CACHE.get(key)
    if cached is not None and cached[0] is topology:
        return cached[1]
    value = compute()
    if len(_PAYLOAD_CACHE) > 65536:
        _PAYLOAD_CACHE.clear()
    _PAYLOAD_CACHE[key] = (topology, value)
    return value


def _domain_link_payload(
    topology: Topology,
    group: Tuple[int, ...],
    rank_map: Mapping[int, int],
) -> tuple:
    return _cached_payload('links', topology, tuple(group), rank_map,
                           lambda: _domain_link_payload_uncached(topology, group, rank_map))


def _domain_link_payload_uncached(
    topology: Topology,
    group: Tuple[int, ...],
    rank_map: Mapping[int, int],
) -> tuple:
    group_set = set(group)
    return tuple(
        sorted(
            (
                rank_map[key.src_rank],
                rank_map[key.dst_rank],
                edge.max_channels,
                _curve_payload(edge.performance),
            )
            for key, edge in topology.links.items()
            if key.src_rank in group_set and key.dst_rank in group_set
        )
    )


def _resource_payload(
    topology: Topology,
    node: PlanNode,
    rank_map: Mapping[int, int],
) -> tuple:
    return _cached_payload('resources', topology,
                           (tuple(node.communication_group), tuple(sorted(node.shared_resource_ids))), rank_map,
                           lambda: _resource_payload_uncached(topology, node, rank_map))


def _resource_payload_uncached(
    topology: Topology,
    node: PlanNode,
    rank_map: Mapping[int, int],
) -> tuple:
    group = set(node.communication_group)
    values = []
    for resource_id in node.shared_resource_ids:
        resource = topology.shared_resources.get(resource_id)
        if resource is None:
            return (("missing", resource_id),)
        domain_members = tuple(
            sorted(
                (
                    rank_map[key.src_rank],
                    rank_map[key.dst_rank],
                )
                for key in resource.member_links
                if key.src_rank in group and key.dst_rank in group
            )
        )
        external_members = sum(
            key.src_rank not in group or key.dst_rank not in group
            for key in resource.member_links
        )
        values.append(
            (
                resource.max_channels,
                _curve_payload(resource.performance),
                domain_members,
                external_members,
                len(resource.member_links),
            )
        )
    return tuple(sorted(values))


_AUTOMORPHISMS: Dict[tuple, tuple] = {}
_MAX_AUTOMORPHISMS = 16


def _domain_automorphisms(topology: Topology, group: Tuple[int, ...]) -> tuple:
    """Position permutations of ``group`` that preserve its domain.

    ``p[i]`` is the image position of ``group[i]``; the identity comes first.
    A permutation must preserve every link attribute the route MILP reads
    (channels, performance, shared-resource capacity and performance). These
    are only candidate maps: template membership is still decided by the
    explicit payload check, so a missing or extra candidate never affects
    soundness, only how many units share a template (SyCCL likewise reuses
    the symmetry of its sketch expansion instead of a general search).
    """
    key = (id(topology), group)
    cached = _AUTOMORPHISMS.get(key)
    if cached is not None and cached[0] is topology:
        return cached[1]
    identity = tuple(range(len(group)))
    result = [identity]
    if 2 <= len(group) <= 16:
        import networkx as nx
        from networkx.algorithms.isomorphism import DiGraphMatcher
        index = {rank: position for position, rank in enumerate(group)}
        graph = nx.DiGraph()
        graph.add_nodes_from(identity)
        for link, edge in topology.links.items():
            if link.src_rank in index and link.dst_rank in index:
                resources = tuple(sorted(
                    (topology.shared_resources[name].max_channels,
                     _curve_payload(topology.shared_resources[name].performance))
                    for name in edge.resource_ids if name in topology.shared_resources))
                graph.add_edge(index[link.src_rank], index[link.dst_rank],
                               label=(edge.max_channels, _curve_payload(edge.performance), resources))
        matcher = DiGraphMatcher(graph, graph, edge_match=lambda left, right: left['label'] == right['label'])
        for mapping in matcher.isomorphisms_iter():
            permutation = tuple(mapping[position] for position in identity)
            if permutation != identity:
                result.append(permutation)
            if len(result) >= _MAX_AUTOMORPHISMS:
                break
    if len(_AUTOMORPHISMS) > 4096:
        _AUTOMORPHISMS.clear()
    _AUTOMORPHISMS[key] = (topology, tuple(result))
    return tuple(result)


def _canonical_rank_map(
    representative_group: Tuple[int, ...],
    representative_permutation: tuple,
    member_group: Tuple[int, ...],
    member_permutation: tuple,
) -> Optional[Dict[int, int]]:
    if len(representative_group) != len(member_group):
        return None
    inverse = {position: member_group[index] for index, position in enumerate(member_permutation)}
    return {rank: inverse[representative_permutation[index]] for index, rank in enumerate(representative_group)}


def _mapped_collective(
    collective: CollectiveSpec,
    rank_map: Mapping[int, int],
) -> tuple:
    root = collective.root
    if root is not None:
        if root not in rank_map:
            return ("unmapped_root",)
        root = rank_map[root]
    return (
        collective.kind.value,
        collective.datatype,
        collective.reduction_op,
        root,
        collective.inplace,
    )


def _basic_roles_match(
    representative: RoutingUnit,
    member: RoutingUnit,
    rank_map: Mapping[int, int],
) -> bool:
    source = representative.node.local_collective
    target = member.node.local_collective
    if (
        representative.node.stage_id != member.node.stage_id
        or source.kind is not target.kind
        or source.datatype != target.datatype
        or source.reduction_op != target.reduction_op
        or source.inplace != target.inplace
        or bool(representative.node.dual_of_node_id)
        != bool(member.node.dual_of_node_id)
    ):
        return False
    if source.root is None:
        if target.root is not None:
            return False
    elif source.root not in rank_map or rank_map[source.root] != target.root:
        return False
    source_roles = tuple(
        sorted(
            (
                rank_map.get(demand.root_rank, -1),
                rank_map.get(demand.required_leaf_rank, -1),
                demand.reduction_dual,
            )
            for demand in representative.demands
        )
    )
    target_roles = tuple(
        sorted(
            (
                demand.root_rank,
                demand.required_leaf_rank,
                demand.reduction_dual,
            )
            for demand in member.demands
        )
    )
    return source_roles == target_roles


def _projected_interface(
    interface: StageInterface,
    contributors: FrozenSet[int],
) -> tuple:
    return tuple(
        (slot, values)
        for slot, values in interface.values.items()
        if values and values <= contributors
    )


def _mapped_offset(
    offset: int,
    values: FrozenSet[int],
    contributor_map: Mapping[int, int],
    position_map: Mapping[int, int],
) -> int:
    if offset in position_map:
        return position_map[offset]
    if len(values) == 1 and offset in values and offset in contributor_map:
        return contributor_map[offset]
    if offset in contributor_map:
        return contributor_map[offset]
    if len(position_map) == 1:
        source_position, target_position = next(iter(position_map.items()))
        return offset + target_position - source_position
    return offset


def _interface_payload(
    interface: StageInterface,
    contributors: FrozenSet[int],
    rank_map: Mapping[int, int],
    contributor_map: Mapping[int, int],
    position_map: Mapping[int, int],
) -> tuple:
    values = []
    for slot, members in _projected_interface(interface, contributors):
        if slot.rank not in rank_map or any(
            contributor not in contributor_map for contributor in members
        ):
            return (("unmapped",),)
        values.append(
            (
                rank_map[slot.rank],
                _mapped_offset(
                    slot.offset,
                    members,
                    contributor_map,
                    position_map,
                ),
                tuple(sorted(contributor_map[item] for item in members)),
            )
        )
    return tuple(sorted(values))


def _mapped_links(
    links: FrozenSet[LinkKey],
    rank_map: Mapping[int, int],
) -> tuple:
    if any(
        link.src_rank not in rank_map or link.dst_rank not in rank_map
        for link in links
    ):
        return ((-1, -1),)
    return tuple(
        sorted(
            (rank_map[link.src_rank], rank_map[link.dst_rank])
            for link in links
        )
    )


def _mapped_forbidden(
    forbidden: Tuple[ForbiddenTransfer, ...],
    rank_map: Mapping[int, int],
    contributor_map: Mapping[int, int],
) -> tuple:
    if any(
        item.slice_id not in contributor_map
        or item.src_rank not in rank_map
        or item.dst_rank not in rank_map
        for item in forbidden
    ):
        return ((-1, -1, -1, -1),)
    return tuple(
        sorted(
            (
                contributor_map[item.slice_id],
                rank_map[item.src_rank],
                rank_map[item.dst_rank],
                item.stage_id,
            )
            for item in forbidden
        )
    )


def _demand_payload(
    demand: TransferDemand,
    rank_map: Mapping[int, int],
    contributor_map: Mapping[int, int],
    position_map: Mapping[int, int],
) -> tuple:
    required_ranks = {demand.root_rank, demand.required_leaf_rank}
    required_ranks.update(
        rank for path in demand.candidate_paths for rank in path
    )
    if (
        not required_ranks <= set(rank_map)
        or demand.logical_position not in position_map
        or not demand.contributors <= set(contributor_map)
        or not demand.member_slice_ids <= set(contributor_map)
    ):
        return (("unmapped",),)
    return (
        demand.stage_id,
        rank_map[demand.root_rank],
        rank_map[demand.required_leaf_rank],
        position_map[demand.logical_position],
        tuple(sorted(contributor_map[item] for item in demand.contributors)),
        tuple(
            sorted(contributor_map[item] for item in demand.member_slice_ids)
        ),
        _mapped_links(demand.allowed_links, rank_map),
        _mapped_links(demand.legal_links, rank_map),
        _mapped_forbidden(
            demand.forbidden_members,
            rank_map,
            contributor_map,
        ),
        tuple(
            sorted(
                tuple(rank_map[rank] for rank in path)
                for path in demand.candidate_paths
            )
        ),
        demand.reduction_dual,
    )


def _candidate_edge_payload(
    problem: SolverProblem,
    unit: RoutingUnit,
    rank_map: Mapping[int, int],
) -> tuple:
    unit_links = frozenset(
        (link.src_rank, link.dst_rank)
        for demand in unit.demands
        for link in demand.legal_links
    )
    values = []
    for edge in problem.candidate_edges:
        if (edge.src_rank, edge.dst_rank) not in unit_links:
            continue
        if edge.src_rank not in rank_map or edge.dst_rank not in rank_map:
            return ((-1, -1, -1),)
        values.append(
            (
                rank_map[edge.src_rank],
                rank_map[edge.dst_rank],
                edge.channel,
            )
        )
    return tuple(sorted(values))


def _unit_payload(
    unit: RoutingUnit,
    problem: SolverProblem,
    rank_map: Mapping[int, int],
    contributor_map: Mapping[int, int],
    position_map: Mapping[int, int],
) -> tuple:
    contributors = _unit_contributors(unit)
    node = unit.node
    if any(rank not in rank_map for rank in node.communication_group):
        return (("unmapped_group",),)
    return (
        node.stage_id,
        _mapped_collective(node.local_collective, rank_map),
        tuple(sorted(rank_map[rank] for rank in node.communication_group)),
        _interface_payload(
            node.logical_input,
            contributors,
            rank_map,
            contributor_map,
            position_map,
        ),
        _interface_payload(
            node.logical_output,
            contributors,
            rank_map,
            contributor_map,
            position_map,
        ),
        _mapped_links(node.allowed_links, rank_map),
        bool(node.dual_of_node_id),
        tuple(
            sorted(
                _demand_payload(
                    demand,
                    rank_map,
                    contributor_map,
                    position_map,
                )
                for demand in unit.demands
            )
        ),
        problem.slice_size_bytes,
        problem.restrictions,
        _candidate_edge_payload(problem, unit, rank_map),
    )


# Equivalence is decided by the explicit rank/contributor/position maps:
# a member joins a template only when every input the route MILP reads
# (domain links with channels and performance, the shared resources on
# those links with capacity, performance and domain members, the demands
# with their candidate paths and restrictions) is equal under the map.
# That is a constructive isomorphism of the subproblem, so a separate
# canonical signature of the whole domain context adds no soundness.  The
# old pure-Python canonical search cost ~4 s per 8-rank domain and 69 s of
# a 2x8 synthesis; SyCCL likewise reuses its expansion mapping (shift test)
# instead of searching for isomorphisms.


def _demand_paths_are_legal(demand: TransferDemand) -> bool:
    allowed = {
        (link.src_rank, link.dst_rank) for link in demand.allowed_links
    }
    legal = {(link.src_rank, link.dst_rank) for link in demand.legal_links}
    forbidden = {
        (item.slice_id, item.src_rank, item.dst_rank)
        for item in demand.forbidden_members
        if item.stage_id == demand.stage_id
    }
    for path in demand.candidate_paths:
        for src, dst in zip(path, path[1:]):
            if (src, dst) not in allowed or (src, dst) not in legal:
                return False
            physical = demand.physical_link(src, dst)
            if any(
                (slice_id, physical[0], physical[1]) in forbidden
                for slice_id in demand.member_slice_ids
            ):
                return False
    return True


def _member_is_legal(
    representative: RoutingUnit,
    representative_problem: SolverProblem,
    member: RoutingUnit,
    member_problem: SolverProblem,
    rank_map: Mapping[int, int],
    contributor_map: Mapping[int, int],
    position_map: Mapping[int, int],
) -> bool:
    if (
        len(rank_map) != len(set(rank_map.values()))
        or len(contributor_map) != len(set(contributor_map.values()))
        or len(position_map) != len(set(position_map.values()))
    ):
        return False
    if _unit_payload(
        representative,
        representative_problem,
        rank_map,
        contributor_map,
        position_map,
    ) != _unit_payload(
        member,
        member_problem,
        {rank: rank for rank in member.node.communication_group},
        {value: value for value in _unit_contributors(member)},
        {value: value for value in _unit_positions(member)},
    ):
        return False
    return all(_demand_paths_are_legal(demand) for demand in member.demands)


def _template_member(
    representative: RoutingUnit,
    representative_problem: SolverProblem,
    member: RoutingUnit,
    member_problem: SolverProblem,
    rank_map: Optional[Dict[int, int]] = None,
) -> Optional[TemplateMember]:
    # A candidate rank map (from domain automorphisms) is only a proposal;
    # every check below holds it to exact equality of the MILP inputs.
    if rank_map is None:
        rank_map = _rank_map(representative, member)
    position_map = _position_map(representative, member)
    if rank_map is None or position_map is None:
        return None
    if not _basic_roles_match(representative, member, rank_map):
        return None
    if _domain_link_payload(
        representative_problem.topology,
        representative.node.communication_group,
        rank_map,
    ) != _domain_link_payload(
        member_problem.topology,
        member.node.communication_group,
        {rank: rank for rank in member.node.communication_group},
    ):
        return None
    if _resource_payload(
        representative_problem.topology,
        representative.node,
        rank_map,
    ) != _resource_payload(
        member_problem.topology,
        member.node,
        {rank: rank for rank in member.node.communication_group},
    ):
        return None
    contributor_map = _contributor_map(
        representative,
        representative_problem,
        member,
        member_problem,
        rank_map,
        position_map,
    )
    if contributor_map is None or not _member_is_legal(
        representative,
        representative_problem,
        member,
        member_problem,
        rank_map,
        contributor_map,
        position_map,
    ):
        return None
    return TemplateMember(
        unit_id=member.unit_id,
        node_id=member.node.node_id,
        rank_map=tuple(sorted(rank_map.items())),
        contributor_map=tuple(sorted(contributor_map.items())),
        logical_position_map=tuple(sorted(position_map.items())),
    )


def _identity_member(unit: RoutingUnit) -> TemplateMember:
    return TemplateMember(
        unit_id=unit.unit_id,
        node_id=unit.node.node_id,
        rank_map=tuple((rank, rank) for rank in unit.node.communication_group),
        contributor_map=tuple(
            (value, value) for value in sorted(_unit_contributors(unit))
        ),
        logical_position_map=tuple(
            (value, value) for value in _unit_positions(unit)
        ),
    )


def _exact_signature(
    unit: RoutingUnit,
    problem: SolverProblem,
    planning_mode: PlanningMode,
) -> str:
    return _signature_digest(_signature_payload(unit, problem, planning_mode))


def _signature_digest(payload: tuple) -> str:
    mode, domain_links, resources, unit = payload
    return sha256_json({"planning_mode": mode, "domain_links": domain_links,
                        "resources": resources, "unit": unit})


def _bucket_key(payload: tuple) -> object:
    """The payload itself when hashable (no JSON/SHA per unit), else its digest."""
    try:
        hash(payload)
    except TypeError:
        return _signature_digest(payload)
    return payload


def _canonical_signature(
    unit: RoutingUnit,
    problem: SolverProblem,
    planning_mode: PlanningMode,
    reuse: bool,
) -> Tuple[tuple, tuple]:
    """(payload, permutation): the smallest payload over the domain's
    automorphisms, so units related by a symmetry of their domain share a
    bucket; the permutation maps the unit onto that canonical labeling."""
    group = unit.node.communication_group
    permutations = _domain_automorphisms(problem.topology, group) if reuse else (tuple(range(len(group))),)
    best = None
    for permutation in permutations:
        payload = _signature_payload(unit, problem, planning_mode, permutation)
        if best is None:
            best = (payload, permutation)
            continue
        try:
            smaller = payload < best[0]
        except TypeError:  # mixed element types: fall back to the text order
            smaller = repr(payload) < repr(best[0])
        if smaller:
            best = (payload, permutation)
    return best


def _signature_payload(
    unit: RoutingUnit,
    problem: SolverProblem,
    planning_mode: PlanningMode,
    permutation: Optional[tuple] = None,
) -> tuple:
    group = unit.node.communication_group
    if permutation is None:
        permutation = tuple(range(len(group)))
    rank_map = {rank: permutation[index] for index, rank in enumerate(group)}
    positions = _unit_positions(unit)
    position_map = {value: index for index, value in enumerate(positions)}
    contributors = sorted(_unit_contributors(unit))
    contributor_keys = []
    for contributor in contributors:
        source_rank, logical_position = divmod(
            contributor,
            problem.slice_count,
        )
        contributor_keys.append(
            (
                (
                    0,
                    rank_map[source_rank],
                    position_map.get(logical_position, logical_position),
                )
                if source_rank in rank_map
                else (
                    1,
                    position_map.get(logical_position, logical_position),
                    contributor,
                ),
                contributor,
            )
        )
    contributor_map = {
        contributor: index
        for index, (_, contributor) in enumerate(sorted(contributor_keys))
    }
    # Everything the route MILP reads, under the group's index map: two units
    # with this payload have identical models up to relabeling.
    return (
        planning_mode.value,
        _domain_link_payload(problem.topology, group, rank_map),
        _resource_payload(problem.topology, unit.node, rank_map),
        _unit_payload(unit, problem, rank_map, contributor_map, position_map),
    )


def build_solver_templates(
    problems: Tuple[SolverProblem, ...],
    planning_mode: PlanningMode,
    *, reuse: bool = True,
    library: Optional[Dict[str, list]] = None,
) -> Tuple[SolverTemplate, ...]:
    """Group routing units whose route models are equal up to relabeling.

    Units are bucketed by their exact signature (everything the route MILP
    reads, under the group's index map) and joined to a class only after the
    explicit-map check in ``_template_member`` succeeds.  ``library`` (a dict
    the caller keeps across calls, e.g. across hierarchy plans) holds earlier
    representatives: a unit matching one joins a template with that
    representative and template ID, so a pattern solved for it is reused.
    """
    if not isinstance(planning_mode, PlanningMode):
        raise SemanticError("planning_mode must be a PlanningMode")
    try:
        problems = tuple(problems)
    except TypeError as error:
        raise SemanticError("problems must be iterable") from error
    if not all(isinstance(problem, SolverProblem) for problem in problems):
        raise SemanticError("problems must contain SolverProblem values")
    units = tuple(
        (unit, problem)
        for problem in problems
        for unit in split_routing_units(problem)
    )
    classes = []
    by_signature: Dict[object, list] = {}
    used_ids = set()
    if library is not None:
        used_ids.update(entry["template_id"] for entries in library.values() for entry in entries)
    def candidate(entry, unit, permutation):
        return _canonical_rank_map(entry["representative"].node.communication_group, entry["permutation"],
                                   unit.node.communication_group, permutation)

    for unit, problem in units:
        payload, permutation = _canonical_signature(unit, problem, planning_mode, reuse)
        signature = _bucket_key(payload)
        joined = False
        if reuse:
            for template_class in by_signature.get(signature, ()):
                member = _template_member(
                    template_class["representative"],
                    template_class["problem"],
                    unit,
                    problem,
                    candidate(template_class, unit, permutation),
                )
                if member is not None:
                    template_class["members"].append(member)
                    joined = True
                    break
            if not joined and library is not None:
                for entry in library.get(signature, ()):
                    member = _template_member(
                        entry["representative"], entry["problem"], unit, problem,
                        candidate(entry, unit, permutation),
                    )
                    if member is not None:
                        template_class = {
                            "representative": entry["representative"],
                            "problem": entry["problem"],
                            "permutation": entry["permutation"],
                            "members": [member],
                            "payload": payload,
                            "digest": entry["digest"],
                            "template_id": entry["template_id"],
                        }
                        classes.append(template_class)
                        by_signature.setdefault(signature, []).append(template_class)
                        joined = True
                        break
        if not joined:
            template_class = {
                "representative": unit,
                "problem": problem,
                "permutation": permutation,
                "members": [_identity_member(unit)],
                "payload": payload,
                "key": signature,
                "digest": None,
                "template_id": None,
            }
            classes.append(template_class)
            if reuse:
                by_signature.setdefault(signature, []).append(template_class)
    templates = []
    for index, template_class in enumerate(classes):
        if template_class["digest"] is None:
            # Only new representatives pay for the canonical JSON digest.
            template_class["digest"] = _signature_digest(template_class["payload"])
        exact_signature = template_class["digest"]
        template_id = template_class["template_id"]
        if template_id is None:
            ordinal = index
            template_id = "template-{:08d}-{}".format(ordinal, exact_signature[:16])
            while template_id in used_ids:
                ordinal += len(classes)
                template_id = "template-{:08d}-{}".format(ordinal, exact_signature[:16])
            used_ids.add(template_id)
            if library is not None:
                library.setdefault(template_class["key"], []).append({
                    "representative": template_class["representative"],
                    "problem": template_class["problem"],
                    "permutation": template_class["permutation"],
                    "template_id": template_id,
                    "digest": exact_signature,
                })
        templates.append(
            SolverTemplate(
                template_id=template_id,
                representative=template_class["representative"],
                members=tuple(template_class["members"]),
                exact_signature=exact_signature,
                library_representative=template_class["template_id"] is not None,
            )
        )
    if sum(len(template.members) for template in templates) != len(units):
        raise SemanticError("solver templates changed the routing unit set")
    return tuple(templates)
