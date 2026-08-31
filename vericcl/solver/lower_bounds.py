import heapq
import math
from dataclasses import dataclass
from typing import Dict, FrozenSet, Tuple

from vericcl.errors import SemanticError, SolverUnavailableError
from vericcl.input.models import ResolvedInput
from vericcl.solver.demands import SolverProblem, TransferDemand
from vericcl.solver.gurobi_api import GurobiAdapter
from vericcl.solver.scheduling import (
    available_channel_count,
    curve_duration_us,
    fixed_transfer_duration_us,
    physical_link_key,
)
from vericcl.topology.model import LinkKey, PerformanceCurve, Topology
from vericcl.topology.performance import safe_per_channel_bandwidth


TreeKey = Tuple[int, int, Tuple[int, ...], bool]


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


def _tree_key(demand: TransferDemand) -> TreeKey:
    return (
        demand.root_rank,
        demand.logical_position,
        tuple(sorted(demand.contributors)),
        demand.reduction_dual,
    )


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


def _resource_time_lower_bound(
    problem: SolverProblem,
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
    model.Params.Seed = problem.inputs.solver.solver_seed
    model.Params.Threads = 1
    model.Params.TimeLimit = problem.inputs.solver.per_model_timeout_s
    trees: Dict[TreeKey, list] = {}
    for demand in problem.demands:
        trees.setdefault(_tree_key(demand), []).append(demand)
    usage = {}
    flow = {}
    for tree_index, (_, demands) in enumerate(sorted(trees.items())):
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
                flow[(demand.demand_id, link)] = variable
                model.addConstr(variable <= usage[(tree_index, link)])
            for rank in problem.node.communication_group:
                outgoing = gp.quicksum(
                    flow[(demand.demand_id, link)]
                    for link in demand.legal_links
                    if link.src_rank == rank
                )
                incoming = gp.quicksum(
                    flow[(demand.demand_id, link)]
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
    for tree_index, (_, demands) in enumerate(sorted(trees.items())):
        representative_by_link = {}
        for demand in demands:
            for link in demand.legal_links:
                representative_by_link.setdefault(link, demand)
        for link, demand in representative_by_link.items():
            physical = physical_link_key(
                demand,
                link.src_rank,
                link.dst_rank,
            )
            variable = usage[(tree_index, link)]
            link_terms.setdefault(physical, []).append(variable)
            for resource_id in problem.topology.link(physical).resource_ids:
                resource_terms.setdefault(resource_id, []).append(variable)
    for physical, variables in link_terms.items():
        edge = problem.topology.link(physical)
        channel_limit = min(max_channels, edge.max_channels)
        channel_limit = min(
            [channel_limit]
            + [
                problem.topology.shared_resources[resource_id].max_channels
                for resource_id in edge.resource_ids
            ]
        )
        capacity = _maximum_capacity(
            edge.performance,
            problem.slice_size_bytes,
            channel_limit,
        )
        if math.isfinite(capacity):
            model.addConstr(
                problem.slice_size_bytes * gp.quicksum(variables)
                <= capacity * tau
            )
    for resource_id, variables in resource_terms.items():
        resource = problem.topology.shared_resources[resource_id]
        capacity = _maximum_capacity(
            resource.performance,
            problem.slice_size_bytes,
            min(max_channels, resource.max_channels),
        )
        if math.isfinite(capacity):
            model.addConstr(
                problem.slice_size_bytes * gp.quicksum(variables)
                <= capacity * tau
            )
    model.setObjective(tau, gp.GRB.MINIMIZE)
    model.optimize()
    result = float(tau.X) if model.SolCount > 0 else 0.0
    model.dispose()
    return max(0.0, result)


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
