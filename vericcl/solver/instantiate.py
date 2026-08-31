from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Mapping, Tuple

from vericcl.composer.dual import reverse_allgather_schedule
from vericcl.errors import SemanticError
from vericcl.input.models import ResolvedInput
from vericcl.planner.model import PlanDAG, PlanNode, StageInterface
from vericcl.semantics.atom import Schedule
from vericcl.solver.demands import TransferDemand, build_solver_problem
from vericcl.solver.lower_bounds import route_edge_duration_us
from vericcl.solver.routing import RoutePattern
from vericcl.solver.scheduling import (
    RoutedOperation,
    RoutedTree,
    materialize_route_schedule,
)
from vericcl.solver.templates import (
    RoutingUnit,
    SolverTemplate,
    TemplateMember,
    split_routing_units,
)
from vericcl.topology.model import LinkKey, Topology


@dataclass(frozen=True)
class InstantiationFailure:
    unit_id: str
    node_id: str
    reason: str

    def __post_init__(self) -> None:
        for field, value in (
            ("unit_id", self.unit_id),
            ("node_id", self.node_id),
            ("reason", self.reason),
        ):
            if not isinstance(value, str) or not value:
                raise SemanticError(
                    "instantiation_failure.{} must be a non-empty string".format(
                        field
                    )
                )


@dataclass(frozen=True)
class InstantiationResult:
    node_schedules: Mapping[str, Schedule]
    failures: Tuple[InstantiationFailure, ...]

    def __post_init__(self) -> None:
        try:
            schedules = dict(self.node_schedules)
        except (TypeError, ValueError) as error:
            raise SemanticError(
                "instantiation_result.node_schedules must be a mapping"
            ) from error
        if any(
            not isinstance(node_id, str)
            or not node_id
            or not isinstance(schedule, Schedule)
            for node_id, schedule in schedules.items()
        ):
            raise SemanticError(
                "instantiation_result.node_schedules contains invalid entries"
            )
        failures = tuple(self.failures)
        if not all(isinstance(item, InstantiationFailure) for item in failures):
            raise SemanticError(
                "instantiation_result.failures must contain InstantiationFailure values"
            )
        object.__setattr__(
            self,
            "node_schedules",
            MappingProxyType(dict(sorted(schedules.items()))),
        )
        object.__setattr__(
            self,
            "failures",
            tuple(
                sorted(
                    failures,
                    key=lambda item: (item.node_id, item.unit_id, item.reason),
                )
            ),
        )


class _MemberInstantiationError(Exception):
    pass


@dataclass(frozen=True)
class _MemberRoute:
    tree: RoutedTree
    operations: Tuple[RoutedOperation, ...]
    transfer_ids: Mapping[Tuple[str, LinkKey], str]
    channel_count: int


def _failure(reason: str) -> None:
    raise _MemberInstantiationError(reason)


def _as_map(
    entries: Tuple[Tuple[int, int], ...],
    reason: str,
) -> Mapping[int, int]:
    result = dict(entries)
    if len(result) != len(entries) or len(set(result.values())) != len(result):
        _failure(reason)
    return result


def _mapped_demand(
    representative: TransferDemand,
    unit: RoutingUnit,
    rank_map: Mapping[int, int],
    contributor_map: Mapping[int, int],
    position_map: Mapping[int, int],
) -> TransferDemand:
    try:
        root_rank = rank_map[representative.root_rank]
        leaf_rank = rank_map[representative.required_leaf_rank]
        logical_position = position_map[representative.logical_position]
        contributors = frozenset(
            contributor_map[value] for value in representative.contributors
        )
        members = frozenset(
            contributor_map[value]
            for value in representative.member_slice_ids
        )
    except KeyError:
        _failure("incomplete_template_member_mapping")
    matches = tuple(
        demand
        for demand in unit.demands
        if demand.root_rank == root_rank
        and demand.required_leaf_rank == leaf_rank
        and demand.logical_position == logical_position
        and demand.contributors == contributors
        and demand.member_slice_ids == members
        and demand.reduction_dual == representative.reduction_dual
    )
    if len(matches) != 1:
        _failure("mapped_demand_is_not_unique")
    return matches[0]


def _physical_link(demand: TransferDemand, link: LinkKey) -> LinkKey:
    return LinkKey(*demand.physical_link(link.src_rank, link.dst_rank))


def _hits_forbidden(demand: TransferDemand, physical: LinkKey) -> bool:
    return any(
        item.stage_id == demand.stage_id
        and item.slice_id in demand.member_slice_ids
        and item.src_rank == physical.src_rank
        and item.dst_rank == physical.dst_rank
        for item in demand.forbidden_members
    )


def _validate_mapped_path(
    demand: TransferDemand,
    path: Tuple[LinkKey, ...],
    node: PlanNode,
    topology: Topology,
) -> None:
    ranks = (path[0].src_rank,) + tuple(link.dst_rank for link in path)
    if (
        ranks[0] != demand.root_rank
        or ranks[-1] != demand.required_leaf_rank
        or len(ranks) != len(set(ranks))
        or any(rank not in node.communication_group for rank in ranks)
        or any(
            first.dst_rank != second.src_rank
            for first, second in zip(path, path[1:])
        )
    ):
        _failure("mapped_route_has_invalid_geometry")
    physical_path = tuple(_physical_link(demand, link) for link in path)
    if any(
        _hits_forbidden(demand, physical) for physical in physical_path
    ):
        _failure("mapped_route_hits_forbidden_transfer")
    for link, physical in zip(path, physical_path):
        if physical not in topology.links:
            _failure("mapped_route_missing_topology_edge")
        if physical not in node.allowed_links:
            _failure("mapped_route_outside_plan_node_domain")
        if link not in demand.allowed_links:
            _failure("mapped_route_outside_demand_domain")
        if link not in demand.legal_links:
            _failure("mapped_route_outside_legal_domain")
        edge = topology.link(physical)
        if any(
            resource_id not in node.shared_resource_ids
            or resource_id not in topology.shared_resources
            or physical
            not in topology.shared_resources[resource_id].member_links
            for resource_id in edge.resource_ids
        ):
            _failure("mapped_route_has_invalid_resource_membership")
    if ranks not in demand.candidate_paths:
        _failure("mapped_route_outside_candidate_path_domain")


def _instantiate_member(
    template: SolverTemplate,
    member: TemplateMember,
    unit: RoutingUnit,
    pattern: RoutePattern,
    inputs: ResolvedInput,
    topology: Topology,
) -> _MemberRoute:
    if pattern.template_id != template.template_id:
        _failure("route_pattern_template_mismatch")
    if pattern.channel_count > inputs.solver.max_channels:
        _failure("route_pattern_channel_count_exceeds_limit")
    rank_map = _as_map(member.rank_map, "invalid_rank_mapping")
    contributor_map = _as_map(
        member.contributor_map,
        "invalid_contributor_mapping",
    )
    position_map = _as_map(
        member.logical_position_map,
        "invalid_logical_position_mapping",
    )
    representative_paths = dict(pattern.member_paths)
    representative_demands = {
        demand.demand_id: demand
        for demand in template.representative.demands
    }
    if set(representative_paths) != set(representative_demands):
        _failure("route_pattern_demand_set_mismatch")
    mapped_paths = []
    target_demands = []
    for demand_id in sorted(representative_demands):
        representative = representative_demands[demand_id]
        target = _mapped_demand(
            representative,
            unit,
            rank_map,
            contributor_map,
            position_map,
        )
        try:
            path = tuple(
                LinkKey(rank_map[src], rank_map[dst])
                for src, dst in representative_paths[demand_id]
            )
        except KeyError:
            _failure("incomplete_rank_mapping")
        _validate_mapped_path(target, path, unit.node, topology)
        mapped_paths.append((target.demand_id, path))
        target_demands.append(target)
    try:
        mapped_selected = {
            LinkKey(rank_map[src], rank_map[dst])
            for src, dst in pattern.selected_edges
        }
    except KeyError:
        _failure("incomplete_rank_mapping")
    used_edges = {
        link for _, path in mapped_paths for link in path
    }
    if mapped_selected != used_edges:
        _failure("mapped_selected_edge_set_mismatch")
    roots = {demand.root_rank for demand in target_demands}
    positions = {demand.logical_position for demand in target_demands}
    contributors = {demand.contributors for demand in target_demands}
    reduction_flags = {demand.reduction_dual for demand in target_demands}
    if not (
        len(roots) == len(positions) == len(contributors) == len(reduction_flags) == 1
    ):
        _failure("mapped_member_crosses_routing_unit_boundary")
    tree = RoutedTree(
        route_id=member.unit_id,
        root_rank=next(iter(roots)),
        logical_position=next(iter(positions)),
        contributors=next(iter(contributors)),
        reduction_dual=next(iter(reduction_flags)),
        demands=tuple(target_demands),
        selected_paths=tuple(mapped_paths),
    )
    demand_by_edge: Dict[LinkKey, TransferDemand] = {}
    for demand, (_, path) in zip(target_demands, mapped_paths):
        for link in path:
            demand_by_edge.setdefault(link, demand)
    parent_ranks = set()
    for link in sorted(used_edges):
        if link.dst_rank in parent_ranks:
            _failure("mapped_route_has_multiple_parents")
        parent_ranks.add(link.dst_rank)
    ready_times = {tree.root_rank: 0.0}
    operations = []
    pending = set(used_edges)
    while pending:
        ready = sorted(
            link for link in pending if link.src_rank in ready_times
        )
        if not ready:
            _failure("mapped_route_is_disconnected")
        for link in ready:
            start = ready_times[link.src_rank]
            duration = route_edge_duration_us(
                inputs,
                topology,
                demand_by_edge[link],
                link,
                pattern.channel_count,
            )
            end = start + duration
            ready_times[link.dst_rank] = end
            operations.append(
                RoutedOperation(
                    route_id=member.unit_id,
                    link=link,
                    channel=0,
                    st_time=start,
                    ed_time=end,
                    resource_slots=(),
                )
            )
            pending.remove(link)
    transfer_ids = {
        (member.unit_id, operation.link): (
            "{}-route-{}-r{:08d}-r{:08d}".format(
                member.node_id,
                member.unit_id,
                operation.link.src_rank,
                operation.link.dst_rank,
            )
        )
        for operation in operations
    }
    return _MemberRoute(
        tree=tree,
        operations=tuple(operations),
        transfer_ids=transfer_ids,
        channel_count=pattern.channel_count,
    )


def _empty_schedule(
    node: PlanNode,
    inputs: ResolvedInput,
    topology: Topology,
    restrictions: Tuple[str, ...],
) -> Schedule:
    return materialize_route_schedule(
        node=node,
        trees=(),
        operations=(),
        transfer_ids={},
        schedule_id="{}-routes".format(node.node_id),
        rank_count=topology.rank_count,
        slice_count=inputs.hyperparameters.slice_count,
        slice_size_bytes=inputs.hyperparameters.slice_size_bytes,
        backend="route_pattern",
        channel_count=1,
        restrictions=restrictions,
        routing_only=True,
        include_resource_order=False,
        include_final_metadata=True,
    )


def _node_schedule(
    node: PlanNode,
    routes: Tuple[_MemberRoute, ...],
    inputs: ResolvedInput,
    topology: Topology,
    restrictions: Tuple[str, ...],
) -> Schedule:
    if not routes:
        return _empty_schedule(node, inputs, topology, restrictions)
    channel_counts = {route.channel_count for route in routes}
    if len(channel_counts) != 1:
        raise SemanticError(
            "route patterns for one node must use one channel count"
        )
    trees = tuple(route.tree for route in routes)
    reduction_flags = {tree.reduction_dual for tree in trees}
    if len(reduction_flags) != 1:
        raise SemanticError("one plan node mixes reduction and send routes")
    transfer_ids = {
        key: transfer_id
        for route in routes
        for key, transfer_id in route.transfer_ids.items()
    }
    virtual = materialize_route_schedule(
        node=node,
        trees=trees,
        operations=tuple(
            operation for route in routes for operation in route.operations
        ),
        transfer_ids=transfer_ids,
        schedule_id="{}-routes".format(node.node_id),
        rank_count=topology.rank_count,
        slice_count=inputs.hyperparameters.slice_count,
        slice_size_bytes=inputs.hyperparameters.slice_size_bytes,
        backend="route_pattern",
        channel_count=next(iter(channel_counts)),
        restrictions=restrictions,
        routing_only=True,
        include_resource_order=False,
        include_final_metadata=not next(iter(reduction_flags)),
    )
    if not next(iter(reduction_flags)):
        return virtual
    available = {
        (tree.root_rank, tree.contributors) for tree in trees
    }
    target_values = {
        slot: contributors
        for slot, contributors in node.logical_output.values.items()
        if (slot.rank, contributors) in available
    }
    if not target_values:
        raise SemanticError("reduction routes do not produce a target interface")
    reduced = reverse_allgather_schedule(
        virtual,
        node.local_collective,
        StageInterface(target_values),
    )
    metadata = dict(reduced.metadata)
    metadata.update(
        {
            "backend": "route_pattern",
            "channel_count": next(iter(channel_counts)),
            "restrictions": restrictions,
            "routing_only": True,
        }
    )
    return Schedule(
        schedule_id=reduced.schedule_id,
        transfers=reduced.transfers,
        final_state_ids=reduced.final_state_ids,
        rank_count=reduced.rank_count,
        slice_count=reduced.slice_count,
        slice_size_bytes=reduced.slice_size_bytes,
        metadata=metadata,
    )


def instantiate_route_patterns(
    plan: PlanDAG,
    templates: Tuple[SolverTemplate, ...],
    patterns: Mapping[str, RoutePattern],
    inputs: ResolvedInput,
    topology: Topology,
) -> InstantiationResult:
    if not isinstance(plan, PlanDAG):
        raise SemanticError("plan must be a PlanDAG")
    if not isinstance(inputs, ResolvedInput):
        raise SemanticError("inputs must be a ResolvedInput")
    if not isinstance(topology, Topology):
        raise SemanticError("topology must be a Topology")
    if (
        inputs.rank_count != topology.rank_count
        or inputs.rank_count != plan.rank_count
        or inputs.hyperparameters.slice_count != plan.slice_count
        or inputs.collective != plan.collective
    ):
        raise SemanticError("plan, inputs, and topology dimensions must agree")
    try:
        templates = tuple(templates)
        patterns = dict(patterns)
    except (TypeError, ValueError) as error:
        raise SemanticError("templates and patterns must be iterable") from error
    if not all(isinstance(template, SolverTemplate) for template in templates):
        raise SemanticError("templates must contain SolverTemplate values")
    if not all(
        isinstance(template_id, str)
        and isinstance(pattern, RoutePattern)
        for template_id, pattern in patterns.items()
    ):
        raise SemanticError("patterns must map template IDs to RoutePattern values")
    template_ids = tuple(template.template_id for template in templates)
    if len(template_ids) != len(set(template_ids)):
        raise SemanticError("template IDs must be unique")
    if not set(patterns) <= set(template_ids):
        raise SemanticError("route pattern references an unknown template")

    problems = {
        node.node_id: build_solver_problem(node, inputs, topology)
        for node in plan.nodes
    }
    units = {
        (node_id, unit.unit_id): unit
        for node_id, problem in problems.items()
        for unit in split_routing_units(problem)
    }
    routes_by_node: Dict[str, list] = {
        node.node_id: [] for node in plan.nodes
    }
    failures = []
    seen_members = set()
    for template in sorted(templates, key=lambda item: item.template_id):
        pattern = patterns.get(template.template_id)
        for member in sorted(
            template.members,
            key=lambda item: (item.node_id, item.unit_id),
        ):
            member_key = (member.node_id, member.unit_id)
            if member_key in seen_members:
                raise SemanticError("template members must be globally unique")
            seen_members.add(member_key)
            if pattern is None:
                failures.append(
                    InstantiationFailure(
                        member.unit_id,
                        member.node_id,
                        "route_pattern_missing",
                    )
                )
                continue
            unit = units.get(member_key)
            if unit is None:
                failures.append(
                    InstantiationFailure(
                        member.unit_id,
                        member.node_id,
                        "routing_unit_missing",
                    )
                )
                continue
            try:
                route = _instantiate_member(
                    template,
                    member,
                    unit,
                    pattern,
                    inputs,
                    topology,
                )
            except _MemberInstantiationError as error:
                failures.append(
                    InstantiationFailure(
                        member.unit_id,
                        member.node_id,
                        str(error),
                    )
                )
                continue
            routes_by_node[member.node_id].append(route)

    schedules = {}
    for node in plan.nodes:
        problem = problems[node.node_id]
        schedules[node.node_id] = _node_schedule(
            node,
            tuple(
                sorted(
                    routes_by_node[node.node_id],
                    key=lambda route: route.tree.route_id,
                )
            ),
            inputs,
            topology,
            problem.restrictions,
        )
    return InstantiationResult(schedules, tuple(failures))
