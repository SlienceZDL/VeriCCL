from typing import Dict, FrozenSet, List, Mapping, Tuple

from vericcl.errors import SemanticError
from vericcl.semantics.atom import Atom, PathStage, Symbol, Transfer
from vericcl.solver.demands import CandidateEdge, SolverProblem, TransferDemand
from vericcl.topology.model import LinkKey, PerformanceCurve
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


def reconstruct_send_transfer(
    *,
    transfer_id: str,
    root_rank: int,
    src_rank: int,
    dst_rank: int,
    parent_by_rank: Mapping[int, int],
    ready_time_by_edge: Mapping[LinkKey, float],
    channel: int,
    stage_id: int,
    member_slice_ids: FrozenSet[int],
    slice_size_bytes: int,
    st_time: float,
    ed_time: float,
    predecessor_ids: FrozenSet[str],
) -> Transfer:
    parents = dict(parent_by_rank)
    ready_times = dict(ready_time_by_edge)
    path = []
    destination = dst_rank
    visited = set()
    while destination != root_rank:
        if destination in visited:
            raise SemanticError("semantic tree path contains a cycle")
        visited.add(destination)
        if destination not in parents:
            raise SemanticError("semantic tree path does not reach its root")
        source = parents[destination]
        edge = LinkKey(source, destination)
        if edge not in ready_times:
            raise SemanticError("semantic tree edge has no ready time")
        path.append(
            Symbol(
                src_rank=source,
                dst_rank=destination,
                ready_time=ready_times[edge],
            )
        )
        destination = source
    symbols = tuple(reversed(path))
    if not symbols or (
        symbols[-1].src_rank,
        symbols[-1].dst_rank,
    ) != (src_rank, dst_rank):
        raise SemanticError("semantic tree path does not end with its transfer")
    atoms = tuple(
        Atom(
            slice_id=slice_id,
            slice_size_bytes=slice_size_bytes,
            path=(PathStage(stage_id, "SEND", symbols),),
            st_time=st_time,
            ed_time=ed_time,
        )
        for slice_id in sorted(member_slice_ids)
    )
    return Transfer(
        transfer_id=transfer_id,
        kind="SEND",
        src_rank=src_rank,
        dst_rank=dst_rank,
        channel=channel,
        stage_id=stage_id,
        member_slice_ids=member_slice_ids,
        atoms=atoms,
        st_time=st_time,
        ed_time=ed_time,
        predecessor_ids=predecessor_ids,
    )


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
