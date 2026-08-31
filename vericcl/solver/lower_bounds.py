import heapq
import math
from dataclasses import dataclass
from typing import Dict, FrozenSet, Tuple

from vericcl.errors import SemanticError, SolverUnavailableError
from vericcl.input.models import ResolvedInput
from vericcl.planner.model import PlanningMode
from vericcl.solver.demands import SolverProblem, TransferDemand
from vericcl.solver.gurobi_api import GurobiAdapter
from vericcl.solver.scheduling import (
    available_channel_count,
    curve_duration_us,
    fixed_transfer_duration_us,
    physical_link_key,
)
from vericcl.solver.templates import (
    RoutingUnit,
    SolverTemplate,
    TemplateMember,
    build_solver_templates,
    split_routing_units,
)
from vericcl.topology.model import LinkKey, PerformanceCurve, Topology
from vericcl.topology.performance import safe_per_channel_bandwidth


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticError("{} must be a number".format(field))
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise SemanticError("{} must be finite and non-negative".format(field))
    return result


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SemanticError("{} must be a positive integer".format(field))
    return value


def route_edge_duration_us(
    inputs: ResolvedInput,
    topology: Topology,
    demand: TransferDemand,
    link: LinkKey,
    channel_count: int,
) -> float:
    if not isinstance(inputs, ResolvedInput):
        raise SemanticError("inputs must be a ResolvedInput")
    if not isinstance(topology, Topology):
        raise SemanticError("topology must be a Topology")
    if not isinstance(demand, TransferDemand):
        raise SemanticError("demand must be a TransferDemand")
    if not isinstance(link, LinkKey):
        raise SemanticError("link must be a LinkKey")
    channels = _positive_integer(channel_count, "channel_count")
    physical = LinkKey(*demand.physical_link(link.src_rank, link.dst_rank))
    edge = topology.link(physical)
    limits = [channels, edge.max_channels]
    limits.extend(
        topology.shared_resources[resource_id].max_channels
        for resource_id in edge.resource_ids
    )
    concurrency = min(limits)
    durations = [
        curve_duration_us(
            edge.performance,
            inputs.hyperparameters.slice_size_bytes,
            concurrency,
        )
    ]
    durations.extend(
        curve_duration_us(
            topology.shared_resources[resource_id].performance,
            inputs.hyperparameters.slice_size_bytes,
            concurrency,
        )
        for resource_id in edge.resource_ids
    )
    return max(durations)


@dataclass(frozen=True)
class LowerBound:
    resource_us: float
    dependency_us: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resource_us",
            _number(self.resource_us, "lower_bound.resource_us"),
        )
        object.__setattr__(
            self,
            "dependency_us",
            _number(self.dependency_us, "lower_bound.dependency_us"),
        )

    @property
    def total_us(self) -> float:
        return max(self.resource_us, self.dependency_us)


def _fastest_edge_duration(
    problem: SolverProblem,
    demand: TransferDemand,
    link: LinkKey,
    max_channels: int,
) -> float:
    available = available_channel_count(
        problem,
        demand,
        link.src_rank,
        link.dst_rank,
        max_channels,
    )
    if available < 1:
        return math.inf
    return min(
        fixed_transfer_duration_us(
            problem,
            demand,
            link.src_rank,
            link.dst_rank,
            concurrency,
        )
        for concurrency in range(1, available + 1)
    )


def _shortest_dependency(
    problem: SolverProblem,
    demand: TransferDemand,
    max_channels: int,
) -> float:
    adjacency: Dict[int, list] = {}
    for link in demand.legal_links:
        duration = _fastest_edge_duration(
            problem,
            demand,
            link,
            max_channels,
        )
        if math.isfinite(duration):
            adjacency.setdefault(link.src_rank, []).append(
                (link.dst_rank, duration)
            )
    pending = [(0.0, demand.root_rank)]
    distances = {demand.root_rank: 0.0}
    while pending:
        distance, rank = heapq.heappop(pending)
        if distance != distances[rank]:
            continue
        if rank == demand.required_leaf_rank:
            return distance
        for destination, duration in adjacency.get(rank, ()):
            candidate = distance + duration
            if candidate < distances.get(destination, math.inf):
                distances[destination] = candidate
                heapq.heappush(pending, (candidate, destination))
    raise SemanticError(
        "demand {} has no legal dependency path".format(demand.demand_id)
    )


def dependency_time_lower_bound(
    problem: SolverProblem,
    max_channels: int,
) -> float:
    if not isinstance(problem, SolverProblem):
        raise SemanticError("problem must be a SolverProblem")
    channels = _positive_integer(max_channels, "max_channels")
    if problem.infeasible_demand_ids:
        raise SemanticError("cannot bound an infeasible solver problem")
    return max(
        (
            _shortest_dependency(problem, demand, channels)
            for demand in problem.demands
        ),
        default=0.0,
    )


def _maximum_capacity(
    curve: PerformanceCurve,
    slice_size_bytes: int,
    max_channels: int,
) -> float:
    if curve.is_calibrated:
        return max(
            concurrency
            * safe_per_channel_bandwidth(curve, concurrency)
            for concurrency in range(1, max_channels + 1)
        )
    return max(
        concurrency
        * slice_size_bytes
        / curve_duration_us(curve, slice_size_bytes, concurrency)
        if curve_duration_us(curve, slice_size_bytes, concurrency) > 0.0
        else math.inf
        for concurrency in range(1, max_channels + 1)
    )


@dataclass(frozen=True)
class _CompressedTreeClass:
    template_id: str
    unit: RoutingUnit
    problem: SolverProblem
    total_bytes: int


def _mapped_member_demands(
    template: SolverTemplate,
    member: TemplateMember,
    unit: RoutingUnit,
) -> Tuple[TransferDemand, ...]:
    rank_map = dict(member.rank_map)
    contributor_map = dict(member.contributor_map)
    position_map = dict(member.logical_position_map)
    mapped = []
    for representative in template.representative.demands:
        try:
            root_rank = rank_map[representative.root_rank]
            leaf_rank = rank_map[representative.required_leaf_rank]
            logical_position = position_map[
                representative.logical_position
            ]
            contributors = frozenset(
                contributor_map[value]
                for value in representative.contributors
            )
            members = frozenset(
                contributor_map[value]
                for value in representative.member_slice_ids
            )
        except KeyError as error:
            raise SemanticError(
                "lower-bound template member mapping is incomplete"
            ) from error
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
            raise SemanticError(
                "lower-bound template member demand mapping is not unique"
            )
        mapped.append(matches[0])
    if len(mapped) != len(unit.demands) or len(set(mapped)) != len(mapped):
        raise SemanticError(
            "lower-bound template member mapping changed the demand set"
        )
    return tuple(mapped)


def _physical_path(
    demand: TransferDemand,
    path: Tuple[int, ...],
) -> tuple:
    return tuple(
        demand.physical_link(src_rank, dst_rank)
        for src_rank, dst_rank in zip(path, path[1:])
    )


def _member_resource_signature(
    template: SolverTemplate,
    member: TemplateMember,
    unit: RoutingUnit,
    problem: SolverProblem,
) -> tuple:
    demands = _mapped_member_demands(template, member, unit)
    demand_values = []
    for demand in demands:
        links = []
        for logical in sorted(demand.legal_links):
            physical = physical_link_key(
                demand,
                logical.src_rank,
                logical.dst_rank,
            )
            edge = problem.topology.link(physical)
            links.append(
                (
                    logical.src_rank,
                    logical.dst_rank,
                    physical.src_rank,
                    physical.dst_rank,
                    tuple(edge.resource_ids),
                )
            )
        demand_values.append(
            (
                demand.root_rank,
                demand.required_leaf_rank,
                demand.reduction_dual,
                tuple(links),
                tuple(
                    sorted(
                        _physical_path(demand, path)
                        for path in demand.candidate_paths
                    )
                ),
            )
        )
    return (
        template.template_id,
        tuple(unit.node.communication_group),
        tuple(sorted(demand_values)),
    )


def _compressed_tree_classes(
    problems: Tuple[SolverProblem, ...],
) -> Tuple[_CompressedTreeClass, ...]:
    problem_by_node = {}
    node_multiplicity: Dict[str, int] = {}
    for problem in problems:
        node_id = problem.node.node_id
        previous = problem_by_node.get(node_id)
        if previous is not None and previous != problem:
            raise SemanticError(
                "duplicate lower-bound plan node IDs must be equivalent"
            )
        problem_by_node.setdefault(node_id, problem)
        node_multiplicity[node_id] = node_multiplicity.get(node_id, 0) + 1
    unique_problems = tuple(problem_by_node.values())
    units = {
        (problem.node.node_id, unit.unit_id): (unit, problem)
        for problem in unique_problems
        for unit in split_routing_units(problem)
    }
    templates = build_solver_templates(
        unique_problems,
        PlanningMode.DIRECT,
    )
    grouped: Dict[tuple, list] = {}
    for template in sorted(templates, key=lambda value: value.template_id):
        for member in sorted(
            template.members,
            key=lambda value: (value.node_id, value.unit_id),
        ):
            key = (member.node_id, member.unit_id)
            if key not in units:
                raise SemanticError(
                    "lower-bound template member has no routing unit"
                )
            unit, problem = units[key]
            signature = _member_resource_signature(
                template,
                member,
                unit,
                problem,
            )
            grouped.setdefault(signature, []).append(
                (
                    unit,
                    problem,
                    node_multiplicity[member.node_id],
                )
            )
    if sum(
        multiplicity
        for values in grouped.values()
        for _, _, multiplicity in values
    ) != sum(
        len(split_routing_units(problem)) for problem in problems
    ):
        raise SemanticError(
            "compressed lower bound changed the routing unit set"
        )
    return tuple(
        _CompressedTreeClass(
            template_id=signature[0],
            unit=values[0][0],
            problem=values[0][1],
            total_bytes=sum(
                problem.slice_size_bytes * multiplicity
                for _, problem, multiplicity in values
            ),
        )
        for signature, values in sorted(grouped.items())
    )


def _global_resource_time_lower_bound(
    problems: Tuple[SolverProblem, ...],
    max_channels: int,
) -> float:
    try:
        gp = GurobiAdapter.require()
    except SolverUnavailableError:
        return 0.0
    try:
        model = gp.Model("vericcl-resource-lower-bound")
    except gp.GurobiError:
        return 0.0
    model.Params.OutputFlag = 0
    reference = problems[0]
    model.Params.Seed = reference.inputs.solver.solver_seed
    model.Params.Threads = 1
    model.Params.TimeLimit = reference.inputs.solver.per_model_timeout_s
    trees = _compressed_tree_classes(problems)
    usage = {}
    flow = {}
    for tree_index, tree in enumerate(trees):
        problem = tree.problem
        demands = tree.unit.demands
        edges = tuple(
            sorted({link for demand in demands for link in demand.legal_links})
        )
        for link in edges:
            usage[(tree_index, link)] = model.addVar(
                lb=0.0,
                ub=1.0,
                vtype=gp.GRB.CONTINUOUS,
                name="usage-t{:04d}-r{:04d}-r{:04d}".format(
                    tree_index,
                    link.src_rank,
                    link.dst_rank,
                ),
            )
        ordered_demands = sorted(demands, key=lambda item: item.demand_id)
        for demand_index, demand in enumerate(ordered_demands):
            for link in sorted(demand.legal_links):
                variable = model.addVar(
                    lb=0.0,
                    ub=1.0,
                    vtype=gp.GRB.CONTINUOUS,
                    name="flow-t{:04d}-d{:04d}-r{:04d}-r{:04d}".format(
                        tree_index,
                        demand_index,
                        link.src_rank,
                        link.dst_rank,
                    ),
                )
                flow[(tree_index, demand_index, link)] = variable
                model.addConstr(variable <= usage[(tree_index, link)])
            for rank in problem.node.communication_group:
                outgoing = gp.quicksum(
                    flow[(tree_index, demand_index, link)]
                    for link in demand.legal_links
                    if link.src_rank == rank
                )
                incoming = gp.quicksum(
                    flow[(tree_index, demand_index, link)]
                    for link in demand.legal_links
                    if link.dst_rank == rank
                )
                target = 0
                if rank == demand.root_rank:
                    target = 1
                elif rank == demand.required_leaf_rank:
                    target = -1
                model.addConstr(outgoing - incoming == target)
    tau = model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name="tau-us")
    link_terms: Dict[LinkKey, list] = {}
    resource_terms: Dict[str, list] = {}
    for tree_index, tree in enumerate(trees):
        problem = tree.problem
        demands = tree.unit.demands
        representative_by_link = {}
        for demand in demands:
            for link in demand.legal_links:
                physical = physical_link_key(
                    demand,
                    link.src_rank,
                    link.dst_rank,
                )
                previous = representative_by_link.setdefault(
                    link,
                    (demand, physical),
                )
                if previous[1] != physical:
                    raise SemanticError(
                        "one compressed tree maps a logical link inconsistently"
                    )
        for link, (_, physical) in representative_by_link.items():
            variable = usage[(tree_index, link)]
            term = (variable, tree.total_bytes)
            link_terms.setdefault(physical, []).append(term)
            for resource_id in problem.topology.link(physical).resource_ids:
                resource_terms.setdefault(resource_id, []).append(term)
    for physical, terms in link_terms.items():
        edge = reference.topology.link(physical)
        channel_limit = min(max_channels, edge.max_channels)
        channel_limit = min(
            [channel_limit]
            + [
                reference.topology.shared_resources[resource_id].max_channels
                for resource_id in edge.resource_ids
            ]
        )
        capacity = _maximum_capacity(
            edge.performance,
            reference.slice_size_bytes,
            channel_limit,
        )
        if math.isfinite(capacity):
            model.addConstr(
                gp.quicksum(
                    slice_size_bytes * variable
                    for variable, slice_size_bytes in terms
                )
                <= capacity * tau
            )
    for resource_id, terms in resource_terms.items():
        resource = reference.topology.shared_resources[resource_id]
        capacity = _maximum_capacity(
            resource.performance,
            reference.slice_size_bytes,
            min(max_channels, resource.max_channels),
        )
        if math.isfinite(capacity):
            model.addConstr(
                gp.quicksum(
                    slice_size_bytes * variable
                    for variable, slice_size_bytes in terms
                )
                <= capacity * tau
            )
    model.setObjective(tau, gp.GRB.MINIMIZE)
    try:
        model.optimize()
        result = float(tau.X) if model.SolCount > 0 else 0.0
    finally:
        model.dispose()
    return max(0.0, result)


def _resource_time_lower_bound(
    problem: SolverProblem,
    max_channels: int,
) -> float:
    return _global_resource_time_lower_bound((problem,), max_channels)


def throughput_time_lower_bound(
    problem: SolverProblem,
    max_channels: int,
) -> LowerBound:
    if not isinstance(problem, SolverProblem):
        raise SemanticError("problem must be a SolverProblem")
    channels = _positive_integer(max_channels, "max_channels")
    return LowerBound(
        resource_us=_resource_time_lower_bound(problem, channels),
        dependency_us=dependency_time_lower_bound(problem, channels),
    )


def global_throughput_time_lower_bound(
    problems: Tuple[SolverProblem, ...],
    max_channels: int,
) -> LowerBound:
    try:
        values = tuple(problems)
    except TypeError as error:
        raise SemanticError("problems must be iterable") from error
    if not values or not all(
        isinstance(problem, SolverProblem) for problem in values
    ):
        raise SemanticError("problems must contain SolverProblem values")
    channels = _positive_integer(max_channels, "max_channels")
    reference = values[0]
    if any(
        problem.topology != reference.topology
        or problem.slice_size_bytes != reference.slice_size_bytes
        for problem in values
    ):
        raise SemanticError(
            "global lower bound problems must share topology and slice size"
        )
    return LowerBound(
        resource_us=_global_resource_time_lower_bound(values, channels),
        dependency_us=max(
            dependency_time_lower_bound(problem, channels)
            for problem in values
        ),
    )
