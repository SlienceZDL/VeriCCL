import math
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Mapping, Optional, Tuple

from vericcl.errors import SemanticError
from vericcl.planner.model import PlanNode
from vericcl.semantics.atom import Atom, PathStage, Schedule, Symbol, Transfer
from vericcl.solver.demands import CandidateEdge, SolverProblem, TransferDemand
from vericcl.topology.model import LaneKey, LinkKey, PerformanceCurve, Topology
from vericcl.topology.performance import (
    safe_per_channel_bandwidth,
    transfer_duration_us,
)


NUMERICAL_TOLERANCE = 1e-6
TreeKey = Tuple[int, int, Tuple[int, ...], bool]
BatchShape = Tuple[
    int,
    bool,
    Tuple[Tuple[int, Tuple[Tuple[int, ...], ...], Tuple[LinkKey, ...]], ...],
]


def _tree_key(demand: TransferDemand) -> TreeKey:
    return (
        demand.root_rank,
        demand.logical_position,
        tuple(sorted(demand.contributors)),
        demand.reduction_dual,
    )


def demand_batch_assignments(
    problem: SolverProblem,
    channel_count: int,
) -> Dict[str, int]:
    by_tree: Dict[TreeKey, List[TransferDemand]] = {}
    for demand in problem.demands:
        by_tree.setdefault(_tree_key(demand), []).append(demand)
    by_shape: Dict[BatchShape, List[TreeKey]] = {}
    for tree_key, demands in by_tree.items():
        shape = (
            tree_key[0],
            tree_key[3],
            tuple(
                sorted(
                    (
                        demand.required_leaf_rank,
                        demand.candidate_paths,
                        tuple(sorted(demand.legal_links)),
                    )
                    for demand in demands
                )
            ),
        )
        by_shape.setdefault(shape, []).append(tree_key)
    assignments = {}
    next_batch = 0
    for shape in sorted(by_shape):
        trees = sorted(by_shape[shape])
        batch_size = channel_count if problem.inputs.strategies.batching else 1
        for index, tree_key in enumerate(trees):
            batch_id = next_batch + index // batch_size
            for demand in by_tree[tree_key]:
                assignments[demand.demand_id] = batch_id
        next_batch += (len(trees) + batch_size - 1) // batch_size
    return assignments


def physical_link_key(
    demand: TransferDemand,
    src_rank: int,
    dst_rank: int,
) -> LinkKey:
    return LinkKey(*demand.physical_link(src_rank, dst_rank))


def available_channel_count(
    problem: SolverProblem,
    demand: TransferDemand,
    src_rank: int,
    dst_rank: int,
    requested: int,
) -> int:
    physical = physical_link_key(demand, src_rank, dst_rank)
    edge = problem.topology.link(physical)
    limits = [requested, edge.max_channels]
    limits.extend(
        problem.topology.shared_resources[resource_id].max_channels
        for resource_id in edge.resource_ids
    )
    available = sum(
        CandidateEdge(src_rank, dst_rank, channel)
        in problem.candidate_edges
        for channel in range(min(limits))
    )
    return min(min(limits), available)


def curve_duration_us(
    curve: PerformanceCurve,
    slice_size_bytes: int,
    concurrency: int,
) -> float:
    if curve.is_calibrated:
        bandwidth = safe_per_channel_bandwidth(curve, concurrency)
        return curve.alpha_us + slice_size_bytes / bandwidth
    return curve.alpha_us + concurrency * curve.beta_effective_us


def fixed_transfer_duration_us(
    problem: SolverProblem,
    demand: TransferDemand,
    src_rank: int,
    dst_rank: int,
    concurrency: int,
) -> float:
    physical = physical_link_key(demand, src_rank, dst_rank)
    edge = problem.topology.link(physical)
    durations = [
        transfer_duration_us(
            edge,
            problem.slice_size_bytes,
            concurrency,
        )
    ]
    durations.extend(
        curve_duration_us(
            problem.topology.shared_resources[resource_id].performance,
            problem.slice_size_bytes,
            concurrency,
        )
        for resource_id in edge.resource_ids
    )
    return max(durations)


def fixed_topology_transfer_duration_us(
    topology: Topology,
    link: LinkKey,
    slice_size_bytes: int,
    channel_count: int,
) -> float:
    """Return the conservative fixed-K duration for one physical transfer."""
    if not isinstance(topology, Topology):
        raise SemanticError("topology must be a Topology")
    if not isinstance(link, LinkKey):
        raise SemanticError("link must be a LinkKey")
    if (
        isinstance(slice_size_bytes, bool)
        or not isinstance(slice_size_bytes, int)
        or slice_size_bytes < 1
    ):
        raise SemanticError("slice_size_bytes must be a positive integer")
    if (
        isinstance(channel_count, bool)
        or not isinstance(channel_count, int)
        or channel_count < 1
    ):
        raise SemanticError("channel_count must be a positive integer")
    edge = topology.link(link)
    durations = [
        curve_duration_us(
            edge.performance,
            slice_size_bytes,
            min(channel_count, edge.max_channels),
        )
    ]
    durations.extend(
        curve_duration_us(
            topology.shared_resources[resource_id].performance,
            slice_size_bytes,
            min(
                channel_count,
                topology.shared_resources[resource_id].max_channels,
            ),
        )
        for resource_id in edge.resource_ids
    )
    return max(durations)


@dataclass(frozen=True)
class RoutedTree:
    route_id: str
    root_rank: int
    logical_position: int
    contributors: FrozenSet[int]
    reduction_dual: bool
    demands: Tuple[TransferDemand, ...]
    selected_paths: Tuple[Tuple[str, Tuple[LinkKey, ...]], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.route_id, str) or not self.route_id:
            raise SemanticError("routed_tree.route_id must be a non-empty string")
        if isinstance(self.root_rank, bool) or not isinstance(self.root_rank, int):
            raise SemanticError("routed_tree.root_rank must be an integer")
        if self.root_rank < 0:
            raise SemanticError("routed_tree.root_rank must be non-negative")
        if (
            isinstance(self.logical_position, bool)
            or not isinstance(self.logical_position, int)
            or self.logical_position < 0
        ):
            raise SemanticError(
                "routed_tree.logical_position must be a non-negative integer"
            )
        contributors = frozenset(self.contributors)
        if not contributors:
            raise SemanticError("routed_tree.contributors must not be empty")
        object.__setattr__(self, "contributors", contributors)
        if not isinstance(self.reduction_dual, bool):
            raise SemanticError("routed_tree.reduction_dual must be a boolean")
        demands = tuple(self.demands)
        if not demands or not all(
            isinstance(demand, TransferDemand) for demand in demands
        ):
            raise SemanticError(
                "routed_tree.demands must contain TransferDemand values"
            )
        demand_ids = tuple(demand.demand_id for demand in demands)
        if len(demand_ids) != len(set(demand_ids)):
            raise SemanticError("routed_tree demand IDs must be unique")
        object.__setattr__(
            self,
            "demands",
            tuple(sorted(demands, key=lambda demand: demand.demand_id)),
        )
        try:
            paths = tuple(
                (demand_id, tuple(path))
                for demand_id, path in self.selected_paths
            )
        except (TypeError, ValueError) as error:
            raise SemanticError(
                "routed_tree.selected_paths must contain demand/path pairs"
            ) from error
        if (
            len(paths) != len(demand_ids)
            or {demand_id for demand_id, _ in paths} != set(demand_ids)
        ):
            raise SemanticError(
                "routed_tree.selected_paths must cover every demand exactly"
            )
        if any(
            not path or not all(isinstance(link, LinkKey) for link in path)
            for _, path in paths
        ):
            raise SemanticError(
                "routed_tree.selected_paths must contain non-empty LinkKey paths"
            )
        object.__setattr__(
            self,
            "selected_paths",
            tuple(sorted(paths, key=lambda item: item[0])),
        )


@dataclass(frozen=True)
class RoutedOperation:
    route_id: str
    link: LinkKey
    channel: int
    st_time: float
    ed_time: float
    resource_slots: Tuple[Tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.route_id, str) or not self.route_id:
            raise SemanticError(
                "routed_operation.route_id must be a non-empty string"
            )
        if not isinstance(self.link, LinkKey):
            raise SemanticError("routed_operation.link must be a LinkKey")
        if (
            isinstance(self.channel, bool)
            or not isinstance(self.channel, int)
            or self.channel < 0
        ):
            raise SemanticError(
                "routed_operation.channel must be a non-negative integer"
            )
        for field, value in (("st_time", self.st_time), ("ed_time", self.ed_time)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0.0
            ):
                raise SemanticError(
                    "routed_operation.{} must be non-negative".format(field)
                )
        if self.st_time > self.ed_time:
            raise SemanticError(
                "routed_operation.st_time must not exceed ed_time"
            )
        try:
            slots = tuple(sorted(tuple(item) for item in self.resource_slots))
        except (TypeError, ValueError) as error:
            raise SemanticError(
                "routed_operation.resource_slots must contain pairs"
            ) from error
        if any(
            len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            or isinstance(item[1], bool)
            or not isinstance(item[1], int)
            or item[1] < 0
            for item in slots
        ):
            raise SemanticError(
                "routed_operation.resource_slots must contain valid pairs"
            )
        if len({item[0] for item in slots}) != len(slots):
            raise SemanticError(
                "routed_operation.resource_slots resource IDs must be unique"
            )
        object.__setattr__(self, "st_time", float(self.st_time))
        object.__setattr__(self, "ed_time", float(self.ed_time))
        object.__setattr__(self, "resource_slots", slots)


def _final_output_key(rank: int, offset: int) -> str:
    return "r{:08d}-o{:08d}".format(rank, offset)


def materialize_route_schedule(
    *,
    node: PlanNode,
    trees: Tuple[RoutedTree, ...],
    operations: Tuple[RoutedOperation, ...],
    transfer_ids: Mapping[Tuple[str, LinkKey], str],
    schedule_id: str,
    rank_count: int,
    slice_count: int,
    slice_size_bytes: int,
    backend: str,
    channel_count: int,
    restrictions: Tuple[str, ...],
    routing_only: bool,
    include_resource_order: bool,
    include_final_metadata: bool,
    extra_metadata: Optional[Mapping[str, object]] = None,
) -> Schedule:
    """Build Schedule semantics from selected routes without solver variables."""
    if not isinstance(node, PlanNode):
        raise SemanticError("node must be a PlanNode")
    trees = tuple(trees)
    operations = tuple(operations)
    tree_by_id = {tree.route_id: tree for tree in trees}
    if len(tree_by_id) != len(trees):
        raise SemanticError("routed tree IDs must be unique")
    operation_by_key = {
        (operation.route_id, operation.link): operation
        for operation in operations
    }
    if len(operation_by_key) != len(operations):
        raise SemanticError("routed operations must be unique per route edge")
    if any(operation.route_id not in tree_by_id for operation in operations):
        raise SemanticError("routed operation references an unknown tree")
    identifiers = dict(transfer_ids)
    if set(identifiers) != set(operation_by_key):
        raise SemanticError("transfer IDs must cover every routed operation")
    if len(set(identifiers.values())) != len(identifiers):
        raise SemanticError("materialized transfer IDs must be unique")

    members_by_key: Dict[Tuple[str, LinkKey], set] = {}
    parents: Dict[Tuple[str, int], int] = {}
    used_keys = set()
    for tree in trees:
        local_operations = {
            link
            for route_id, link in operation_by_key
            if route_id == tree.route_id
        }
        for link in local_operations:
            if link.dst_rank == tree.root_rank:
                raise SemanticError("materialized route enters its root")
            parent_key = (tree.route_id, link.dst_rank)
            if parent_key in parents:
                raise SemanticError("materialized route gives a rank multiple parents")
            parents[parent_key] = link.src_rank
        tree_paths = dict(tree.selected_paths)
        for demand in tree.demands:
            path = tree_paths[demand.demand_id]
            ranks = (path[0].src_rank,) + tuple(link.dst_rank for link in path)
            if (
                path[0].src_rank != demand.root_rank
                or path[-1].dst_rank != demand.required_leaf_rank
                or any(
                    first.dst_rank != second.src_rank
                    for first, second in zip(path, path[1:])
                )
                or len(ranks) != len(set(ranks))
            ):
                raise SemanticError("materialized demand path is invalid")
            for link in path:
                key = (tree.route_id, link)
                if key not in operation_by_key:
                    raise SemanticError(
                        "materialized demand path omits an operation"
                    )
                members_by_key.setdefault(key, set()).update(
                    demand.member_slice_ids
                )
                used_keys.add(key)
        if local_operations != {
            link for route_id, link in used_keys if route_id == tree.route_id
        }:
            raise SemanticError("materialized route contains an unused operation")
    if used_keys != set(operation_by_key):
        raise SemanticError("materialized route operation set is incomplete")

    semantic_predecessors = {key: set() for key in operation_by_key}
    predecessors = {key: set() for key in operation_by_key}
    for key, operation in operation_by_key.items():
        parent_rank = parents.get((operation.route_id, operation.link.src_rank))
        if parent_rank is None:
            continue
        parent_key = (
            operation.route_id,
            LinkKey(parent_rank, operation.link.src_rank),
        )
        predecessor_id = identifiers[parent_key]
        semantic_predecessors[key].add(predecessor_id)
        predecessors[key].add(predecessor_id)
    if include_resource_order:
        lane_groups: Dict[LaneKey, list] = {}
        resource_groups: Dict[Tuple[str, int], list] = {}
        for key, operation in operation_by_key.items():
            lane_groups.setdefault(
                LaneKey(
                    operation.link.src_rank,
                    operation.link.dst_rank,
                    operation.channel,
                ),
                [],
            ).append((key, operation))
            for resource_id, slot in operation.resource_slots:
                resource_groups.setdefault((resource_id, slot), []).append(
                    (key, operation)
                )
        for entries in tuple(lane_groups.values()) + tuple(
            resource_groups.values()
        ):
            ordered = sorted(
                entries,
                key=lambda item: (item[1].st_time, item[0]),
            )
            for first, second in zip(ordered, ordered[1:]):
                predecessors[second[0]].add(identifiers[first[0]])

    def tree_path(key: Tuple[str, LinkKey]) -> Tuple[RoutedOperation, ...]:
        operation = operation_by_key[key]
        tree = tree_by_id[operation.route_id]
        path = []
        destination = operation.link.dst_rank
        while destination != tree.root_rank:
            source = parents[(operation.route_id, destination)]
            parent = operation_by_key[
                (operation.route_id, LinkKey(source, destination))
            ]
            path.append(parent)
            destination = source
        return tuple(reversed(path))

    transfers = []
    path_prefixes = {}
    for key in sorted(operation_by_key):
        operation = operation_by_key[key]
        tree = tree_by_id[operation.route_id]
        path = tree_path(key)
        symbols = []
        for item in path:
            parent_rank = parents.get((item.route_id, item.link.src_rank))
            ready_time = 0.0
            if parent_rank is not None:
                ready_time = operation_by_key[
                    (item.route_id, LinkKey(parent_rank, item.link.src_rank))
                ].ed_time
            symbols.append(
                Symbol(
                    src_rank=item.link.src_rank,
                    dst_rank=item.link.dst_rank,
                    ready_time=ready_time,
                )
            )
        members = frozenset(members_by_key[key])
        atoms = tuple(
            Atom(
                slice_id=slice_id,
                slice_size_bytes=slice_size_bytes,
                path=(
                    PathStage(
                        stage_id=node.stage_id,
                        operator="SEND",
                        symbols=tuple(symbols),
                    ),
                ),
                st_time=operation.st_time,
                ed_time=operation.ed_time,
            )
            for slice_id in sorted(members)
        )
        transfer_id = identifiers[key]
        transfers.append(
            Transfer(
                transfer_id=transfer_id,
                kind="SEND",
                src_rank=operation.link.src_rank,
                dst_rank=operation.link.dst_rank,
                channel=operation.channel,
                stage_id=node.stage_id,
                member_slice_ids=members,
                atoms=atoms,
                st_time=operation.st_time,
                ed_time=operation.ed_time,
                predecessor_ids=frozenset(predecessors[key]),
            )
        )
        path_prefixes[transfer_id] = {
            slice_id: tuple(
                (symbol.src_rank, symbol.dst_rank) for symbol in symbols
            )
            for slice_id in sorted(members)
        }

    metadata = {
        "backend": backend,
        "channel_count": channel_count,
        "path_scope": "stage_suffix",
        "path_roots": {
            identifiers[key]: tree_by_id[key[0]].root_rank
            for key in sorted(operation_by_key)
        },
        "reduction_dual": bool(trees) and all(
            tree.reduction_dual for tree in trees
        ),
        "restrictions": tuple(restrictions),
        "semantic_contributors": {
            identifiers[key]: tuple(sorted(members_by_key[key]))
            for key in sorted(operation_by_key)
        },
        "semantic_predecessors": {
            identifiers[key]: tuple(sorted(semantic_predecessors[key]))
            for key in sorted(operation_by_key)
        },
        "tree_contributors": {
            identifiers[key]: tuple(
                sorted(tree_by_id[key[0]].contributors)
            )
            for key in sorted(operation_by_key)
        },
        "resource_slots": {
            identifiers[key]: dict(operation_by_key[key].resource_slots)
            for key in sorted(operation_by_key)
        },
    }
    if routing_only:
        metadata["routing_only"] = True
        metadata["path_prefixes"] = path_prefixes
    final_outputs = {}
    final_dependencies = {}
    if include_final_metadata:
        for slot, contributors in node.logical_output.values.items():
            key = _final_output_key(slot.rank, slot.offset)
            matches = tuple(
                sorted(
                    identifiers[(tree.route_id, operation.link)]
                    for tree in trees
                    if tree.contributors == contributors
                    and not tree.reduction_dual
                    for operation in operations
                    if operation.route_id == tree.route_id
                    and operation.link.dst_rank == slot.rank
                )
            )
            passthrough = any(
                input_slot.rank == slot.rank
                and input_contributors == contributors
                for input_slot, input_contributors in (
                    node.logical_input.values.items()
                )
            )
            if not matches and not passthrough:
                continue
            final_outputs[key] = tuple(sorted(contributors))
            final_dependencies[key] = matches
        metadata["final_outputs"] = final_outputs
        metadata["final_dependencies"] = final_dependencies
    if extra_metadata:
        metadata.update(dict(extra_metadata))
    final_state_keys = (
        tuple(sorted(final_outputs))
        if include_final_metadata
        else tuple(
            _final_output_key(slot.rank, slot.offset)
            for slot in node.logical_output.values
        )
    )
    return Schedule(
        schedule_id=schedule_id,
        transfers=tuple(transfers),
        final_state_ids=tuple(
            "{}-{}".format(node.node_id, key) for key in final_state_keys
        ),
        rank_count=rank_count,
        slice_count=slice_count,
        slice_size_bytes=slice_size_bytes,
        metadata=metadata,
    )
