from typing import Dict, FrozenSet, List, Mapping, Tuple

from vericcl.errors import SemanticError
from vericcl.semantics.atom import Atom, PathStage, Schedule, Symbol, Transfer
from vericcl.solver.demands import CandidateEdge, SolverProblem, TransferDemand
from vericcl.topology.model import LinkKey, PerformanceCurve, Topology
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


def rebuild_scheduled_transfers(
    schedule: Schedule,
    *,
    timings: Mapping[str, tuple[float, float, float]],
    channels: Mapping[str, int],
    semantic_predecessors: Mapping[str, FrozenSet[str]],
    predecessor_ids: Mapping[str, FrozenSet[str]],
    resource_slots: Mapping[str, Mapping[str, int]],
    metadata_updates: Mapping[str, object] | None = None,
) -> Schedule:
    if not isinstance(schedule, Schedule):
        raise SemanticError("schedule must be a Schedule")
    transfer_ids = {
        transfer.transfer_id for transfer in schedule.transfers
    }
    mappings = (
        timings,
        channels,
        semantic_predecessors,
        predecessor_ids,
        resource_slots,
    )
    if any(set(values) != transfer_ids for values in mappings):
        raise SemanticError("rebuilt schedule metadata must cover every transfer")
    operation_ids = {}
    for transfer in schedule.transfers:
        for member in transfer.member_slice_ids:
            key = (
                transfer.stage_id,
                transfer.src_rank,
                transfer.dst_rank,
                member,
            )
            if (
                key in operation_ids
                and operation_ids[key] != transfer.transfer_id
            ):
                raise SemanticError("slice operation is duplicated in the schedule")
            operation_ids[key] = transfer.transfer_id
    rebuilt = []
    for transfer in schedule.transfers:
        transfer_id = transfer.transfer_id
        start, end, _ = timings[transfer_id]
        atoms = []
        for atom in transfer.atoms:
            path = []
            for stage in atom.path:
                symbols = []
                for symbol in stage.symbols:
                    operation_id = operation_ids.get(
                        (
                            stage.stage_id,
                            symbol.src_rank,
                            symbol.dst_rank,
                            atom.slice_id,
                        )
                    )
                    if operation_id is None:
                        raise SemanticError(
                            "atom path operation is missing from the schedule"
                        )
                    symbols.append(
                        Symbol(
                            symbol.src_rank,
                            symbol.dst_rank,
                            timings[operation_id][2],
                        )
                    )
                path.append(
                    PathStage(stage.stage_id, stage.operator, tuple(symbols))
                )
            atoms.append(
                Atom(
                    slice_id=atom.slice_id,
                    slice_size_bytes=atom.slice_size_bytes,
                    path=tuple(path),
                    st_time=start,
                    ed_time=end,
                )
            )
        rebuilt.append(
            Transfer(
                transfer_id=transfer_id,
                kind=transfer.kind,
                src_rank=transfer.src_rank,
                dst_rank=transfer.dst_rank,
                channel=channels[transfer_id],
                stage_id=transfer.stage_id,
                member_slice_ids=transfer.member_slice_ids,
                atoms=tuple(atoms),
                st_time=start,
                ed_time=end,
                predecessor_ids=predecessor_ids[transfer_id],
            )
        )
    metadata = dict(schedule.metadata)
    metadata["semantic_predecessors"] = {
        transfer_id: tuple(sorted(values))
        for transfer_id, values in semantic_predecessors.items()
    }
    metadata["resource_slots"] = {
        transfer_id: dict(values)
        for transfer_id, values in resource_slots.items()
    }
    metadata["timing_recomputed"] = True
    if metadata_updates is not None:
        metadata.update(metadata_updates)
    return Schedule(
        schedule_id=schedule.schedule_id,
        transfers=tuple(sorted(rebuilt, key=lambda item: item.transfer_id)),
        final_state_ids=schedule.final_state_ids,
        rank_count=schedule.rank_count,
        slice_count=schedule.slice_count,
        slice_size_bytes=schedule.slice_size_bytes,
        metadata=metadata,
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


def conservative_transfer_duration_us(
    topology: Topology,
    link: LinkKey,
    slice_size_bytes: int,
    channel_count: int,
) -> float:
    if not isinstance(topology, Topology):
        raise SemanticError("topology must be a Topology")
    if not isinstance(link, LinkKey) or link not in topology.links:
        raise SemanticError("transfer link is absent from the topology")
    if (
        isinstance(channel_count, bool)
        or not isinstance(channel_count, int)
        or channel_count < 1
    ):
        raise SemanticError("channel_count must be a positive integer")
    edge = topology.link(link)
    durations = [
        transfer_duration_us(
            edge,
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
