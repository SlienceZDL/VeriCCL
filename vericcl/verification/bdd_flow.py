from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Tuple

from vericcl.errors import SemanticError
from vericcl.input.models import ResolvedInput
from vericcl.semantics.atom import Schedule
from vericcl.solver.pruning import ranked_simple_paths, retain_shortest_paths
from vericcl.topology.model import LaneKey, LinkKey, Topology
from vericcl.verification.bdd_backend import BDDAnalysisResult, CompactBDD
from vericcl.verification.flow_index import FlowRecord, build_flow_index
from vericcl.verification.model import ValidationStatus


@dataclass(frozen=True)
class FlowReplacementHint:
    source_flow_id: str
    demand_id: str
    candidate_flow_ids: Tuple[str, ...]
    candidate_paths: Mapping[str, Tuple[int, ...]]
    candidate_first_lanes: Mapping[str, LaneKey]
    divergence_rank: int
    waiting_transfer_id: str
    bottleneck_lane: LaneKey
    wait_start_us: float
    wait_end_us: float
    earliest_candidate_start_us: float

    def __post_init__(self) -> None:
        candidates = tuple(sorted(set(self.candidate_flow_ids)))
        paths = {
            key: tuple(value) for key, value in self.candidate_paths.items()
        }
        lanes = dict(self.candidate_first_lanes)
        if set(paths) != set(candidates) or set(lanes) != set(candidates):
            raise SemanticError("candidate flow metadata must cover all candidates")
        object.__setattr__(self, "candidate_flow_ids", candidates)
        object.__setattr__(self, "candidate_paths", MappingProxyType(paths))
        object.__setattr__(self, "candidate_first_lanes", MappingProxyType(lanes))

    @property
    def wait_interval_us(self) -> Tuple[float, float]:
        return self.wait_start_us, self.wait_end_us


@dataclass(frozen=True)
class _Candidate:
    candidate_id: str
    source_flow: FlowRecord
    waiting_transfer_id: str
    bottleneck_lane: LaneKey
    divergence_rank: int
    wait_start_us: float
    wait_end_us: float
    path: Tuple[int, ...]
    first_lane: LaneKey
    earliest_start_us: Optional[float]


def _legal_links(
    flow: FlowRecord,
    topology: Topology,
    inputs: ResolvedInput,
) -> frozenset[LinkKey]:
    forbidden = {
        LinkKey(item.src_rank, item.dst_rank)
        for item in inputs.atom_constraints.forbidden_transfers
        if item.stage_id == flow.stage_id
        and item.slice_id in flow.member_slice_ids
    }
    return frozenset(topology.links) - forbidden


def _candidate_paths(
    flow: FlowRecord,
    divergence_index: int,
    topology: Topology,
    inputs: ResolvedInput,
    path_cache: Optional[Dict[tuple, Tuple[Tuple[int, ...], ...]]] = None,
) -> Tuple[Tuple[int, ...], ...]:
    source = flow.ranks[divergence_index]
    destination = flow.leaf_rank
    legal = _legal_links(flow, topology, inputs)

    def edge_cost(src_rank: int, dst_rank: int) -> float:
        return topology.link(LinkKey(src_rank, dst_rank)).performance.invbw_us

    key = (legal, source, destination, inputs.strategies.shortest_paths)
    if path_cache is not None and key in path_cache:
        paths = path_cache[key]
    else:
        paths = ranked_simple_paths(legal, source, destination, edge_cost, limit=32)
        if inputs.strategies.shortest_paths:
            paths = retain_shortest_paths(paths, edge_cost)
        if path_cache is not None:
            path_cache[key] = paths
    # Prefix exclusion is flow-specific, even when the ranked suffix query is
    # shared.  Preserve filtering after the original top-32/shortest selection.
    prefix_ranks = set(flow.ranks[:divergence_index])
    return tuple(
        path for path in paths if not prefix_ranks.intersection(path[1:])
    )


def _discover_candidates(
    schedule: Schedule,
    topology: Topology,
    inputs: ResolvedInput,
) -> Tuple[_Candidate, ...]:
    index = build_flow_index(schedule)
    transfers = {
        transfer.transfer_id: transfer for transfer in schedule.transfers
    }
    candidates = []
    # Topology, inputs, and lane intervals are immutable during this analysis.
    # Scope caches to this call so changed constraints or schedules cannot leak.
    path_cache: Dict[tuple, Tuple[Tuple[int, ...], ...]] = {}
    idle_cache: Dict[tuple, Optional[float]] = {}
    for flow in index.flows:
        for position in range(flow.comparison_end):
            transfer_id = flow.transfer_ids[position]
            transfer = transfers[transfer_id]
            ready_time = flow.ready_times[position]
            if ready_time >= transfer.st_time:
                continue
            duration = transfer.ed_time - transfer.st_time
            for path_index, path in enumerate(
                _candidate_paths(flow, position, topology, inputs, path_cache)
            ):
                first_link = LinkKey(path[0], path[1])
                channel_count = min(
                    topology.link(first_link).max_channels,
                    inputs.solver.max_channels,
                )
                for channel in range(channel_count):
                    first_lane = LaneKey(path[0], path[1], channel)
                    if first_lane == flow.lanes[position]:
                        continue
                    idle_key = (first_lane, ready_time, transfer.st_time, duration)
                    if idle_key not in idle_cache:
                        idle_cache[idle_key] = index.lane(first_lane).earliest_start(
                            ready_time,
                            transfer.st_time,
                            duration,
                        )
                    earliest = idle_cache[idle_key]
                    candidate_id = (
                        "candidate-{}-{}-p{:04d}-c{:04d}".format(
                            flow.flow_id,
                            transfer_id,
                            path_index,
                            channel,
                        )
                    )
                    candidates.append(
                        _Candidate(
                            candidate_id=candidate_id,
                            source_flow=flow,
                            waiting_transfer_id=transfer_id,
                            bottleneck_lane=flow.lanes[position],
                            divergence_rank=flow.ranks[position],
                            wait_start_us=ready_time,
                            wait_end_us=transfer.st_time,
                            path=path,
                            first_lane=first_lane,
                            earliest_start_us=earliest,
                        )
                    )
    return tuple(sorted(candidates, key=lambda item: item.candidate_id))


def _success(hints: Tuple[FlowReplacementHint, ...], evidence: Mapping[str, object]):
    return BDDAnalysisResult(
        status=ValidationStatus.VALID,
        code="flow_bdd_analysis_complete",
        message="flow congestion BDD analysis completed",
        hints=hints,
        evidence=evidence,
    )


def analyze_flow_congestion(
    schedule: Schedule,
    topology: Topology,
    inputs: ResolvedInput,
    *, backend_name: str = "cudd",
) -> BDDAnalysisResult:
    try:
        CompactBDD({"availability": 1}, backend=backend_name)
        if not isinstance(schedule, Schedule):
            raise SemanticError("schedule must be a Schedule")
        if not isinstance(topology, Topology):
            raise SemanticError("topology must be a Topology")
        if not isinstance(inputs, ResolvedInput):
            raise SemanticError("inputs must be a ResolvedInput")
        if schedule.rank_count != topology.rank_count:
            raise SemanticError("schedule and topology rank counts must agree")
        candidates = _discover_candidates(schedule, topology, inputs)
        if not candidates:
            return _success((), {"candidate_count": 0, "relation_count": 0})

        candidate_ids = {
            value: index
            for index, value in enumerate(
                sorted(item.candidate_id for item in candidates)
            )
        }
        # Candidate ID uniquely determines flow, demand, and first lane.  The
        # projection onto that key is therefore a bijection on compatible rows
        # and commutes with intersection with their idle subset.  Keep metadata
        # in the candidates instead of encoding redundant BDD dimensions.
        backend = CompactBDD({"candidate_flow_id": len(candidate_ids)}, backend=backend_name)
        # Every candidate passed route compatibility, so the compatible set is
        # exactly the bounded domain, including for non-power-of-two sizes.
        compatible = backend.relation(()).complement()
        idle_count = sum(item.earliest_start_us is not None for item in candidates)
        encode_idle = idle_count <= len(candidates) - idle_count
        # Idle and blocked candidates partition the compatible domain.  Encode
        # the smaller side, using bounded BDD complement when blocked is smaller.
        partition = backend.relation(
            (candidate_ids[item.candidate_id],)
            for item in candidates
            if (item.earliest_start_us is not None) == encode_idle
        )
        idle = partition if encode_idle else partition.complement()
        selected_rows = compatible.intersection(idle).tuples()
        selected_candidate_ids = {
            row[0] for row in selected_rows
        }
        selected = tuple(
            item
            for item in candidates
            if candidate_ids[item.candidate_id] in selected_candidate_ids
        )

        grouped: Dict[tuple, list] = {}
        for item in selected:
            key = (
                item.source_flow.flow_id,
                item.source_flow.demand_id,
                item.divergence_rank,
                item.waiting_transfer_id,
                item.bottleneck_lane,
                item.wait_start_us,
                item.wait_end_us,
            )
            grouped.setdefault(key, []).append(item)
        hints = []
        for key, values in sorted(grouped.items(), key=lambda item: item[0]):
            (
                source_flow_id,
                demand_id,
                divergence_rank,
                waiting_transfer_id,
                bottleneck_lane,
                wait_start,
                wait_end,
            ) = key
            hints.append(
                FlowReplacementHint(
                    source_flow_id=source_flow_id,
                    demand_id=demand_id,
                    candidate_flow_ids=tuple(
                        item.candidate_id for item in values
                    ),
                    candidate_paths={
                        item.candidate_id: item.path for item in values
                    },
                    candidate_first_lanes={
                        item.candidate_id: item.first_lane for item in values
                    },
                    divergence_rank=divergence_rank,
                    waiting_transfer_id=waiting_transfer_id,
                    bottleneck_lane=bottleneck_lane,
                    wait_start_us=wait_start,
                    wait_end_us=wait_end,
                    earliest_candidate_start_us=min(
                        item.earliest_start_us
                        for item in values
                        if item.earliest_start_us is not None
                    ),
                )
            )
        return _success(
            tuple(hints),
            {
                "candidate_count": len(candidates),
                "relation_count": len(selected_rows),
                "bdd_variable_count": backend.variable_count,
            },
        )
    except Exception as error:
        return BDDAnalysisResult(
            status=ValidationStatus.ANALYSIS_ERROR,
            code="flow_bdd_analysis_error",
            message="flow congestion BDD analysis failed",
            hints=(),
            evidence={
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
