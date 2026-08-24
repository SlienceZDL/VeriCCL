from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Mapping

from vericcl.errors import SemanticError
from vericcl.semantics.atom import Schedule, Transfer
from vericcl.solver.demands import SolverProblem, TransferDemand
from vericcl.solver.routing import RoutePattern
from vericcl.solver.scheduling import reconstruct_send_transfer
from vericcl.solver.templates import (
    RoutingUnit,
    SolverTemplate,
    TemplateMember,
    split_routing_units,
)
from vericcl.topology.model import LinkKey


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SemanticError("{} must be a non-empty string".format(field))
    return value


@dataclass(frozen=True)
class InstantiationFailure:
    unit_id: str
    node_id: str
    reason: str

    def __post_init__(self) -> None:
        _identifier(self.unit_id, "instantiation_failure.unit_id")
        _identifier(self.node_id, "instantiation_failure.node_id")
        _identifier(self.reason, "instantiation_failure.reason")


@dataclass(frozen=True)
class InstantiationResult:
    node_schedules: Mapping[str, Schedule]
    failures: tuple[InstantiationFailure, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.node_schedules, Mapping):
            raise SemanticError("node_schedules must be a mapping")
        schedules = dict(self.node_schedules)
        if not all(
            isinstance(node_id, str) and isinstance(schedule, Schedule)
            for node_id, schedule in schedules.items()
        ):
            raise SemanticError("node_schedules must map node IDs to Schedule values")
        failures = tuple(self.failures)
        if not all(isinstance(item, InstantiationFailure) for item in failures):
            raise SemanticError(
                "failures must contain InstantiationFailure values"
            )
        object.__setattr__(
            self,
            "node_schedules",
            MappingProxyType(dict(sorted(schedules.items()))),
        )
        object.__setattr__(
            self,
            "failures",
            tuple(sorted(failures, key=lambda item: (item.unit_id, item.reason))),
        )


@dataclass(frozen=True)
class _InstantiatedUnit:
    transfers: tuple[Transfer, ...]
    path_roots: Mapping[str, int]
    semantic_contributors: Mapping[str, tuple[int, ...]]
    semantic_predecessors: Mapping[str, tuple[str, ...]]
    tree_contributors: Mapping[str, tuple[int, ...]]
    resource_slots: Mapping[str, Mapping[str, int]]
    route_template_ids: Mapping[str, str]
    route_unit_ids: Mapping[str, str]


def _unit_slice_ids(unit: RoutingUnit) -> frozenset[int]:
    return frozenset(
        slice_id
        for demand in unit.demands
        for values in (demand.contributors, demand.member_slice_ids)
        for slice_id in values
    )


def _unit_logical_positions(unit: RoutingUnit) -> frozenset[int]:
    return frozenset(demand.logical_position for demand in unit.demands)


def _demand_semantics(
    demand: TransferDemand,
    rank_map: Mapping[int, int],
    contributor_map: Mapping[int, int],
    logical_map: Mapping[int, int],
) -> tuple:
    try:
        return (
            demand.stage_id,
            rank_map[demand.root_rank],
            rank_map[demand.required_leaf_rank],
            logical_map[demand.logical_position],
            tuple(sorted(contributor_map[item] for item in demand.contributors)),
            tuple(
                sorted(contributor_map[item] for item in demand.member_slice_ids)
            ),
            demand.reduction_dual,
        )
    except KeyError as error:
        raise SemanticError("template member mapping is incomplete") from error


def _target_demand_semantics(demand: TransferDemand) -> tuple:
    return (
        demand.stage_id,
        demand.root_rank,
        demand.required_leaf_rank,
        demand.logical_position,
        tuple(sorted(demand.contributors)),
        tuple(sorted(demand.member_slice_ids)),
        demand.reduction_dual,
    )


def _validate_member_semantics(
    template: SolverTemplate,
    member: TemplateMember,
    unit: RoutingUnit,
) -> Mapping[int, int]:
    if member.node_id != unit.node.node_id:
        raise SemanticError("template member node does not match its routing unit")
    rank_map = dict(member.rank_map)
    contributor_map = dict(member.contributor_map)
    logical_map = dict(member.logical_position_map)
    if set(rank_map) != set(template.representative.node.communication_group):
        raise SemanticError("template rank mapping source coverage is invalid")
    if set(rank_map.values()) != set(unit.node.communication_group):
        raise SemanticError("template rank mapping target coverage is invalid")
    if set(contributor_map) != _unit_slice_ids(template.representative):
        raise SemanticError("template contributor mapping source coverage is invalid")
    if set(contributor_map.values()) != _unit_slice_ids(unit):
        raise SemanticError("template contributor mapping target coverage is invalid")
    if set(logical_map) != _unit_logical_positions(template.representative):
        raise SemanticError("template logical mapping source coverage is invalid")
    if set(logical_map.values()) != _unit_logical_positions(unit):
        raise SemanticError("template logical mapping target coverage is invalid")
    mapped = tuple(
        sorted(
            _demand_semantics(
                demand,
                rank_map,
                contributor_map,
                logical_map,
            )
            for demand in template.representative.demands
        )
    )
    target = tuple(
        sorted(_target_demand_semantics(demand) for demand in unit.demands)
    )
    if mapped != target:
        raise SemanticError("template member demand mapping is no longer exact")
    return rank_map


def _path_to_leaf(
    root_rank: int,
    leaf_rank: int,
    parent_by_rank: Mapping[int, int],
) -> tuple[LinkKey, ...]:
    reversed_path = []
    visited = set()
    rank = leaf_rank
    while rank != root_rank:
        if rank in visited:
            raise SemanticError("mapped route contains a cycle")
        visited.add(rank)
        if rank not in parent_by_rank:
            raise SemanticError("mapped route does not reach a required leaf")
        parent = parent_by_rank[rank]
        reversed_path.append(LinkKey(parent, rank))
        rank = parent
    return tuple(reversed(reversed_path))


def _physical_edge(demand: TransferDemand, edge: LinkKey) -> LinkKey:
    return LinkKey(*demand.physical_link(edge.src_rank, edge.dst_rank))


def _mapped_route(
    template: SolverTemplate,
    member: TemplateMember,
    pattern: RoutePattern,
    unit: RoutingUnit,
    problem: SolverProblem,
) -> _InstantiatedUnit:
    if pattern.template_id != template.template_id:
        raise SemanticError("route pattern template ID does not match")
    rank_map = _validate_member_semantics(template, member, unit)
    try:
        edges = tuple(
            sorted(
                LinkKey(rank_map[src_rank], rank_map[dst_rank])
                for src_rank, dst_rank in pattern.parent_edges
            )
        )
        mapped_selected = frozenset(
            LinkKey(rank_map[edge.src_rank], rank_map[edge.dst_rank])
            for edge in pattern.selected_edges
        )
    except KeyError as error:
        raise SemanticError("route pattern rank mapping is incomplete") from error
    root_ranks = {demand.root_rank for demand in unit.demands}
    if len(root_ranks) != 1:
        raise SemanticError("routing unit demands do not share one root")
    root_rank = next(iter(root_ranks))
    parent_by_rank: Dict[int, int] = {}
    for edge in edges:
        if edge.dst_rank == root_rank:
            raise SemanticError("mapped route enters its root")
        if edge.dst_rank in parent_by_rank:
            raise SemanticError("mapped route gives a rank multiple parents")
        parent_by_rank[edge.dst_rank] = edge.src_rank
    physical_edges = frozenset(_physical_edge(unit.demands[0], edge) for edge in edges)
    if mapped_selected != physical_edges:
        raise SemanticError("mapped route physical edges do not match its tree")
    members_by_edge: Dict[LinkKey, set[int]] = {}
    for demand in unit.demands:
        path = _path_to_leaf(
            demand.root_rank,
            demand.required_leaf_rank,
            parent_by_rank,
        )
        for edge in path:
            if edge not in demand.allowed_links or edge not in demand.legal_links:
                raise SemanticError("mapped route uses an illegal or disallowed edge")
            physical = _physical_edge(demand, edge)
            if physical not in problem.topology.links:
                raise SemanticError("mapped route uses a missing physical link")
            if any(
                forbidden.slice_id in demand.member_slice_ids
                and forbidden.src_rank == physical.src_rank
                and forbidden.dst_rank == physical.dst_rank
                and forbidden.stage_id == demand.stage_id
                for forbidden in demand.forbidden_members
            ):
                raise SemanticError("mapped route uses a slice-specific forbidden transfer")
            members_by_edge.setdefault(edge, set()).update(demand.member_slice_ids)
    if set(members_by_edge) != set(edges):
        raise SemanticError("mapped route contains an unused or disconnected edge")
    identifiers = {
        edge: "{}-route-t{:08d}".format(unit.unit_id, index)
        for index, edge in enumerate(edges)
    }
    ready_times = {edge: 0.0 for edge in edges}
    contributors = tuple(sorted(unit.demands[0].contributors))
    transfers = []
    path_roots = {}
    semantic_contributors = {}
    semantic_predecessors = {}
    tree_contributors = {}
    resource_slots = {}
    route_template_ids = {}
    route_unit_ids = {}
    for edge in edges:
        predecessor_ids = frozenset()
        if edge.src_rank != root_rank:
            predecessor_edge = LinkKey(
                parent_by_rank[edge.src_rank],
                edge.src_rank,
            )
            predecessor_ids = frozenset({identifiers[predecessor_edge]})
        transfer_id = identifiers[edge]
        members = frozenset(members_by_edge[edge])
        transfers.append(
            reconstruct_send_transfer(
                transfer_id=transfer_id,
                root_rank=root_rank,
                src_rank=edge.src_rank,
                dst_rank=edge.dst_rank,
                parent_by_rank=parent_by_rank,
                ready_time_by_edge=ready_times,
                channel=0,
                stage_id=unit.demands[0].stage_id,
                member_slice_ids=members,
                slice_size_bytes=problem.slice_size_bytes,
                st_time=0.0,
                ed_time=0.0,
                predecessor_ids=predecessor_ids,
            )
        )
        path_roots[transfer_id] = root_rank
        semantic_contributors[transfer_id] = tuple(sorted(members))
        semantic_predecessors[transfer_id] = tuple(sorted(predecessor_ids))
        tree_contributors[transfer_id] = contributors
        resource_slots[transfer_id] = {}
        route_template_ids[transfer_id] = template.template_id
        route_unit_ids[transfer_id] = unit.unit_id
    return _InstantiatedUnit(
        transfers=tuple(transfers),
        path_roots=path_roots,
        semantic_contributors=semantic_contributors,
        semantic_predecessors=semantic_predecessors,
        tree_contributors=tree_contributors,
        resource_slots=resource_slots,
        route_template_ids=route_template_ids,
        route_unit_ids=route_unit_ids,
    )


def _merge(target: dict, values: Mapping) -> None:
    overlap = set(target) & set(values)
    if overlap:
        raise SemanticError("instantiated transfer IDs are not unique")
    target.update(values)


def instantiate_route_patterns(
    templates: tuple[SolverTemplate, ...],
    patterns: Mapping[str, RoutePattern],
    problems: tuple[SolverProblem, ...],
) -> InstantiationResult:
    try:
        templates = tuple(templates)
        problems = tuple(problems)
    except TypeError as error:
        raise SemanticError("templates and problems must be iterable") from error
    if not all(isinstance(item, SolverTemplate) for item in templates):
        raise SemanticError("templates must contain SolverTemplate values")
    if not isinstance(patterns, Mapping) or not all(
        isinstance(key, str) and isinstance(value, RoutePattern)
        for key, value in patterns.items()
    ):
        raise SemanticError("patterns must map template IDs to RoutePattern values")
    if not all(isinstance(item, SolverProblem) for item in problems):
        raise SemanticError("problems must contain SolverProblem values")
    problem_by_node = {problem.node.node_id: problem for problem in problems}
    if len(problem_by_node) != len(problems):
        raise SemanticError("solver problem node IDs must be unique")
    unit_context = {}
    for problem in problems:
        for unit in split_routing_units(problem):
            if unit.unit_id in unit_context:
                raise SemanticError("routing unit IDs must be unique")
            unit_context[unit.unit_id] = (unit, problem)
    channel_counts = {pattern.channel_count for pattern in patterns.values()}
    if len(channel_counts) > 1:
        raise SemanticError("route patterns must use one channel count")
    channel_count = next(iter(channel_counts), 0)
    transfers_by_node = {node_id: [] for node_id in problem_by_node}
    metadata_by_node = {
        node_id: {
            "path_roots": {},
            "semantic_contributors": {},
            "semantic_predecessors": {},
            "tree_contributors": {},
            "resource_slots": {},
            "route_template_ids": {},
            "route_unit_ids": {},
        }
        for node_id in problem_by_node
    }
    failures = []
    covered_units = set()
    for template in sorted(templates, key=lambda item: item.template_id):
        pattern = patterns.get(template.template_id)
        for member in template.members:
            if member.unit_id in covered_units:
                raise SemanticError("routing unit belongs to multiple templates")
            covered_units.add(member.unit_id)
            context = unit_context.get(member.unit_id)
            if context is None:
                failures.append(
                    InstantiationFailure(
                        member.unit_id,
                        member.node_id,
                        "routing unit is unavailable",
                    )
                )
                continue
            unit, problem = context
            if pattern is None:
                failures.append(
                    InstantiationFailure(
                        member.unit_id,
                        member.node_id,
                        "route pattern is unavailable",
                    )
                )
                continue
            try:
                instantiated = _mapped_route(
                    template,
                    member,
                    pattern,
                    unit,
                    problem,
                )
            except SemanticError as error:
                failures.append(
                    InstantiationFailure(
                        member.unit_id,
                        member.node_id,
                        str(error),
                    )
                )
                continue
            transfers_by_node[member.node_id].extend(instantiated.transfers)
            metadata = metadata_by_node[member.node_id]
            for field in metadata:
                _merge(metadata[field], getattr(instantiated, field))
    for unit_id in sorted(set(unit_context) - covered_units):
        unit, _ = unit_context[unit_id]
        failures.append(
            InstantiationFailure(
                unit_id,
                unit.node.node_id,
                "routing unit is not covered by a template",
            )
        )
    schedules = {}
    for node_id, problem in sorted(problem_by_node.items()):
        metadata = metadata_by_node[node_id]
        transfers = tuple(
            sorted(
                transfers_by_node[node_id],
                key=lambda transfer: transfer.transfer_id,
            )
        )
        schedules[node_id] = Schedule(
            schedule_id="{}-routing-only".format(node_id),
            transfers=transfers,
            final_state_ids=tuple(
                "{}-r{:08d}-o{:08d}".format(node_id, slot.rank, slot.offset)
                for slot in sorted(problem.node.logical_output.values)
            ),
            rank_count=problem.topology.rank_count,
            slice_count=problem.slice_count,
            slice_size_bytes=problem.slice_size_bytes,
            metadata={
                "backend": "template_routing",
                "routing_only": True,
                "channel_count": channel_count,
                "path_scope": "stage_suffix",
                "path_roots": metadata["path_roots"],
                "reduction_dual": problem.reduction_dual,
                "restrictions": problem.restrictions,
                "semantic_contributors": metadata["semantic_contributors"],
                "semantic_predecessors": metadata["semantic_predecessors"],
                "tree_contributors": metadata["tree_contributors"],
                "resource_slots": metadata["resource_slots"],
                "route_template_ids": metadata["route_template_ids"],
                "route_unit_ids": metadata["route_unit_ids"],
            },
        )
    return InstantiationResult(schedules, tuple(failures))
