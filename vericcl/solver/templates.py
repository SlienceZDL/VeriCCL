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
from vericcl.topology.isomorphism import exact_domain_signature
from vericcl.topology.model import LinkKey, PerformanceCurve, Topology


_CHAIN_KINDS = frozenset(
    {
        CollectiveKind.GATHER,
        CollectiveKind.SCATTER,
        CollectiveKind.ALL_TO_ALL,
    }
)
_CANONICALIZATION_LIMIT = (
    "domain isomorphism canonicalization limit exceeded"
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
        unit_ids = tuple(member.unit_id for member in members)
        if len(unit_ids) != len(set(unit_ids)):
            raise SemanticError("solver template unit IDs must be unique")
        if self.representative.unit_id not in set(unit_ids):
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
    )


def _domain_link_payload(
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


def _safe_domain_signature(
    topology: Topology,
    group: Tuple[int, ...],
    cache: Dict[tuple, Optional[str]],
) -> Optional[str]:
    key = (topology.isomorphism_signature, group)
    if key in cache:
        return cache[key]
    try:
        signature = exact_domain_signature(topology, group)
    except SemanticError as error:
        if str(error) != _CANONICALIZATION_LIMIT:
            raise
        signature = None
    cache[key] = signature
    return signature


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
    signature_cache: Dict[tuple, Optional[str]],
) -> Optional[TemplateMember]:
    rank_map = _rank_map(representative, member)
    position_map = _position_map(representative, member)
    if rank_map is None or position_map is None:
        return None
    if not _basic_roles_match(representative, member, rank_map):
        return None
    representative_signature = _safe_domain_signature(
        representative_problem.topology,
        representative.node.communication_group,
        signature_cache,
    )
    member_signature = _safe_domain_signature(
        member_problem.topology,
        member.node.communication_group,
        signature_cache,
    )
    if (
        representative_signature is None
        or member_signature is None
        or representative_signature != member_signature
    ):
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
    signature_cache: Dict[tuple, Optional[str]],
) -> str:
    group = unit.node.communication_group
    rank_map = {rank: index for index, rank in enumerate(group)}
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
    domain_signature = _safe_domain_signature(
        problem.topology,
        group,
        signature_cache,
    )
    payload = {
        "planning_mode": planning_mode.value,
        "domain_signature": domain_signature,
        "domain_links": _domain_link_payload(
            problem.topology,
            group,
            rank_map,
        ),
        "resources": _resource_payload(
            problem.topology,
            unit.node,
            rank_map,
        ),
        "unit": _unit_payload(
            unit,
            problem,
            rank_map,
            contributor_map,
            position_map,
        ),
    }
    if domain_signature is None:
        payload["unproved_unit_id"] = unit.unit_id
        payload["unproved_topology"] = problem.topology.isomorphism_signature
    return sha256_json(payload)


def build_solver_templates(
    problems: Tuple[SolverProblem, ...],
    planning_mode: PlanningMode,
) -> Tuple[SolverTemplate, ...]:
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
    signature_cache: Dict[tuple, Optional[str]] = {}
    classes = []
    for unit, problem in units:
        for template_class in classes:
            member = _template_member(
                template_class["representative"],
                template_class["problem"],
                unit,
                problem,
                signature_cache,
            )
            if member is not None:
                template_class["members"].append(member)
                break
        else:
            classes.append(
                {
                    "representative": unit,
                    "problem": problem,
                    "members": [_identity_member(unit)],
                }
            )
    templates = []
    for index, template_class in enumerate(classes):
        exact_signature = _exact_signature(
            template_class["representative"],
            template_class["problem"],
            planning_mode,
            signature_cache,
        )
        templates.append(
            SolverTemplate(
                template_id="template-{:08d}-{}".format(
                    index,
                    exact_signature[:16],
                ),
                representative=template_class["representative"],
                members=tuple(template_class["members"]),
                exact_signature=exact_signature,
            )
        )
    if sum(len(template.members) for template in templates) != len(units):
        raise SemanticError("solver templates changed the routing unit set")
    return tuple(templates)
