from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Tuple

from vericcl.errors import SemanticError
from vericcl.semantics.atom import Schedule, Transfer
from vericcl.topology.model import (
    LaneKey,
    LinkKey,
    PerformanceCurve,
    Topology,
)
from vericcl.topology.performance import (
    safe_per_channel_bandwidth,
    transfer_duration_us,
)
from vericcl.verification.resource_events import (
    ResourceTimeline,
    build_resource_timelines,
    directed_link_timeline_id,
    shared_resource_timeline_id,
)


_TOLERANCE = 1e-12


def _time(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticError("{} must be a number".format(field))
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise SemanticError(
            "{} must be finite and non-negative".format(field)
        )
    return normalized


@dataclass(frozen=True)
class SimulationEvent:
    time_us: float
    event_type: str
    transfer_id: str
    lane: LaneKey
    link_concurrency: int
    resource_concurrency: Tuple[Tuple[str, int], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "time_us",
            _time(self.time_us, "simulation_event.time_us"),
        )
        if self.event_type not in {"start", "complete"}:
            raise SemanticError("simulation event type is invalid")
        if not isinstance(self.transfer_id, str) or not self.transfer_id:
            raise SemanticError("simulation event transfer ID is invalid")
        if not isinstance(self.lane, LaneKey):
            raise SemanticError("simulation event lane is invalid")
        if (
            isinstance(self.link_concurrency, bool)
            or not isinstance(self.link_concurrency, int)
            or self.link_concurrency < 1
        ):
            raise SemanticError("simulation event link concurrency is invalid")
        values = tuple(sorted(self.resource_concurrency))
        if not all(
            isinstance(resource_id, str)
            and resource_id
            and isinstance(concurrency, int)
            and not isinstance(concurrency, bool)
            and concurrency >= 1
            for resource_id, concurrency in values
        ):
            raise SemanticError(
                "simulation event resource concurrency is invalid"
            )
        object.__setattr__(self, "resource_concurrency", values)


@dataclass(frozen=True)
class SimulationResult:
    events: Tuple[SimulationEvent, ...]
    completion_time_us: float
    start_times: Mapping[str, float]
    end_times: Mapping[str, float]
    semantic_ready_times: Mapping[str, float]
    queue_wait_times: Mapping[str, float]
    timelines: Mapping[str, ResourceTimeline]

    def __post_init__(self) -> None:
        events = tuple(self.events)
        if not all(isinstance(event, SimulationEvent) for event in events):
            raise SemanticError("simulation result events are invalid")
        completion = _time(
            self.completion_time_us,
            "simulation_result.completion_time_us",
        )
        mappings = {}
        for field in (
            "start_times",
            "end_times",
            "semantic_ready_times",
            "queue_wait_times",
        ):
            raw = getattr(self, field)
            if not isinstance(raw, Mapping):
                raise SemanticError(
                    "simulation result {} must be a mapping".format(field)
                )
            values = {
                transfer_id: _time(value, field)
                for transfer_id, value in raw.items()
            }
            if not all(
                isinstance(transfer_id, str) and transfer_id
                for transfer_id in values
            ):
                raise SemanticError(
                    "simulation result {} keys are invalid".format(field)
                )
            mappings[field] = MappingProxyType(dict(sorted(values.items())))
        transfer_ids = set(mappings["start_times"])
        if any(set(values) != transfer_ids for values in mappings.values()):
            raise SemanticError("simulation result transfer mappings differ")
        timelines = dict(self.timelines)
        if not all(
            isinstance(resource_id, str)
            and isinstance(timeline, ResourceTimeline)
            and resource_id == timeline.resource_id
            for resource_id, timeline in timelines.items()
        ):
            raise SemanticError("simulation result timelines are invalid")
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "completion_time_us", completion)
        for field, values in mappings.items():
            object.__setattr__(self, field, values)
        object.__setattr__(
            self,
            "timelines",
            MappingProxyType(dict(sorted(timelines.items()))),
        )


def _semantic_predecessors(schedule: Schedule) -> Mapping[str, frozenset]:
    raw = schedule.metadata.get("semantic_predecessors")
    if raw is None:
        return {
            transfer.transfer_id: transfer.predecessor_ids
            for transfer in schedule.transfers
        }
    if not isinstance(raw, Mapping):
        raise SemanticError("semantic_predecessors must be a mapping")
    expected = {transfer.transfer_id for transfer in schedule.transfers}
    if set(raw) != expected:
        raise SemanticError(
            "semantic_predecessors must cover every transfer"
        )
    result = {key: frozenset(value) for key, value in raw.items()}
    if any(not predecessors <= expected for predecessors in result.values()):
        raise SemanticError("semantic predecessor is missing")
    return result


def _resource_slots(
    schedule: Schedule,
    topology: Topology,
) -> Mapping[str, Mapping[str, int]]:
    raw = schedule.metadata.get("resource_slots", {})
    if not isinstance(raw, Mapping):
        raise SemanticError("resource_slots must be a mapping")
    result = {}
    for transfer in schedule.transfers:
        key = LinkKey(transfer.src_rank, transfer.dst_rank)
        if key not in topology.links:
            raise SemanticError("simulation transfer uses a missing link")
        edge = topology.link(key)
        if transfer.channel >= edge.max_channels:
            raise SemanticError("simulation transfer channel exceeds its link")
        values = raw.get(transfer.transfer_id, {})
        if not isinstance(values, Mapping):
            raise SemanticError("transfer resource slots must be a mapping")
        if set(values) != set(edge.resource_ids):
            raise SemanticError(
                "transfer resource slots do not match its directed link"
            )
        slots = {}
        for resource_id, slot in values.items():
            resource = topology.shared_resources[resource_id]
            if (
                isinstance(slot, bool)
                or not isinstance(slot, int)
                or slot < 0
                or slot >= resource.max_channels
            ):
                raise SemanticError("transfer resource slot is invalid")
            slots[resource_id] = slot
        result[transfer.transfer_id] = MappingProxyType(slots)
    return MappingProxyType(result)


def _curve_duration_us(
    curve: PerformanceCurve,
    slice_size_bytes: int,
    concurrency: int,
) -> float:
    if curve.is_calibrated:
        return curve.alpha_us + slice_size_bytes / (
            safe_per_channel_bandwidth(curve, concurrency)
        )
    return curve.alpha_us + concurrency * curve.beta_effective_us


def _queue_heads(queues, completed):
    return {
        key: next(
            (
                transfer_id
                for transfer_id in queue
                if transfer_id not in completed
            ),
            None,
        )
        for key, queue in queues.items()
    }


def _active_sets(active, transfers, topology):
    by_link = defaultdict(set)
    by_resource = defaultdict(set)
    for transfer_id in active:
        transfer = transfers[transfer_id]
        key = LinkKey(transfer.src_rank, transfer.dst_rank)
        by_link[key].add(transfer_id)
        for resource_id in topology.resources_for(key):
            by_resource[resource_id].add(transfer_id)
    return by_link, by_resource


def _duration(
    transfer: Transfer,
    topology: Topology,
    by_link,
    by_resource,
) -> float:
    key = LinkKey(transfer.src_rank, transfer.dst_rank)
    edge = topology.link(key)
    durations = [
        transfer_duration_us(
            edge,
            transfer.physical_bytes,
            len(by_link[key]),
        )
    ]
    for resource_id in edge.resource_ids:
        resource = topology.shared_resources[resource_id]
        concurrency = len(by_resource[resource_id])
        if concurrency > resource.max_channels:
            raise SemanticError("shared resource concurrency exceeds capacity")
        durations.append(
            _curve_duration_us(
                resource.performance,
                transfer.physical_bytes,
                concurrency,
            )
        )
    return max(durations)


def _timeline_snapshot(active, transfers, topology):
    by_link, by_resource = _active_sets(active, transfers, topology)
    values = {
        directed_link_timeline_id(key): tuple(sorted(by_link[key]))
        for key in topology.links
    }
    values.update(
        {
            shared_resource_timeline_id(resource_id): tuple(
                sorted(by_resource[resource_id])
            )
            for resource_id in topology.shared_resources
        }
    )
    return values


def simulate_schedule(
    schedule: Schedule,
    topology: Topology,
) -> SimulationResult:
    if not isinstance(schedule, Schedule):
        raise SemanticError("schedule must be a Schedule")
    if not isinstance(topology, Topology):
        raise SemanticError("topology must be a Topology")
    if schedule.rank_count != topology.rank_count:
        raise SemanticError("schedule and topology rank counts differ")
    transfers = {
        transfer.transfer_id: transfer for transfer in schedule.transfers
    }
    semantic = _semantic_predecessors(schedule)
    slots = _resource_slots(schedule, topology)
    order_key = lambda transfer_id: (
        transfers[transfer_id].st_time,
        transfers[transfer_id].ed_time,
        transfer_id,
    )
    lane_queues = defaultdict(list)
    resource_queues = defaultdict(list)
    for transfer in schedule.transfers:
        lane = LaneKey(
            transfer.src_rank,
            transfer.dst_rank,
            transfer.channel,
        )
        lane_queues[lane].append(transfer.transfer_id)
        for resource_id, slot in slots[transfer.transfer_id].items():
            resource_queues[(resource_id, slot)].append(
                transfer.transfer_id
            )
    for queue in tuple(lane_queues.values()) + tuple(
        resource_queues.values()
    ):
        queue.sort(key=order_key)

    unfinished = set(transfers)
    completed = set()
    active = {}
    start_times = {}
    end_times = {}
    ready_times = {}
    wait_times = {}
    events = []
    snapshots = []
    current_time = 0.0

    while unfinished or active:
        started = True
        while started:
            started = False
            lane_heads = _queue_heads(lane_queues, completed)
            resource_heads = _queue_heads(resource_queues, completed)
            by_link, by_resource = _active_sets(
                active,
                transfers,
                topology,
            )
            candidates = []
            for transfer_id in unfinished:
                transfer = transfers[transfer_id]
                required_predecessors = (
                    transfer.predecessor_ids | semantic[transfer_id]
                )
                if not required_predecessors <= completed:
                    continue
                lane = LaneKey(
                    transfer.src_rank,
                    transfer.dst_rank,
                    transfer.channel,
                )
                if lane_heads[lane] != transfer_id:
                    continue
                if any(
                    resource_heads[(resource_id, slot)] != transfer_id
                    for resource_id, slot in slots[transfer_id].items()
                ):
                    continue
                key = LinkKey(transfer.src_rank, transfer.dst_rank)
                if len(by_link[key]) >= topology.link(key).max_channels:
                    continue
                if any(
                    len(by_resource[resource_id])
                    >= topology.shared_resources[resource_id].max_channels
                    for resource_id in slots[transfer_id]
                ):
                    continue
                candidates.append(transfer_id)
            if candidates:
                transfer_id = min(candidates, key=order_key)
                transfer = transfers[transfer_id]
                unfinished.remove(transfer_id)
                active[transfer_id] = 1.0
                ready_time = max(
                    (
                        end_times[predecessor]
                        for predecessor in semantic[transfer_id]
                    ),
                    default=0.0,
                )
                start_times[transfer_id] = current_time
                ready_times[transfer_id] = ready_time
                wait_times[transfer_id] = current_time - ready_time
                by_link, by_resource = _active_sets(
                    active,
                    transfers,
                    topology,
                )
                link = LinkKey(transfer.src_rank, transfer.dst_rank)
                events.append(
                    SimulationEvent(
                        current_time,
                        "start",
                        transfer_id,
                        LaneKey(
                            transfer.src_rank,
                            transfer.dst_rank,
                            transfer.channel,
                        ),
                        len(by_link[link]),
                        tuple(
                            (
                                resource_id,
                                len(by_resource[resource_id]),
                            )
                            for resource_id in topology.resources_for(link)
                        ),
                    )
                )
                started = True

        if not active:
            if unfinished:
                raise SemanticError("simulation cannot make progress")
            break
        by_link, by_resource = _active_sets(active, transfers, topology)
        durations = {
            transfer_id: _duration(
                transfers[transfer_id],
                topology,
                by_link,
                by_resource,
            )
            for transfer_id in active
        }
        instant = sorted(
            transfer_id
            for transfer_id, duration in durations.items()
            if duration <= _TOLERANCE
        )
        if instant:
            completed_now = instant
        else:
            delta = min(
                active[transfer_id] * durations[transfer_id]
                for transfer_id in active
            )
            next_time = current_time + delta
            snapshots.append(
                (
                    current_time,
                    next_time,
                    _timeline_snapshot(active, transfers, topology),
                )
            )
            for transfer_id in tuple(active):
                active[transfer_id] = max(
                    0.0,
                    active[transfer_id]
                    - delta / durations[transfer_id],
                )
            current_time = next_time
            completed_now = sorted(
                transfer_id
                for transfer_id, remaining in active.items()
                if remaining <= _TOLERANCE
            )
        completion_by_link, completion_by_resource = _active_sets(
            active,
            transfers,
            topology,
        )
        for transfer_id in completed_now:
            transfer = transfers[transfer_id]
            del active[transfer_id]
            completed.add(transfer_id)
            end_times[transfer_id] = current_time
            link = LinkKey(transfer.src_rank, transfer.dst_rank)
            events.append(
                SimulationEvent(
                    current_time,
                    "complete",
                    transfer_id,
                    LaneKey(
                        transfer.src_rank,
                        transfer.dst_rank,
                        transfer.channel,
                    ),
                    len(completion_by_link[link]),
                    tuple(
                        (
                            resource_id,
                            len(completion_by_resource[resource_id]),
                        )
                        for resource_id in topology.resources_for(link)
                    ),
                )
            )

    return SimulationResult(
        events=tuple(events),
        completion_time_us=current_time,
        start_times=start_times,
        end_times=end_times,
        semantic_ready_times=ready_times,
        queue_wait_times=wait_times,
        timelines=build_resource_timelines(topology, tuple(snapshots)),
    )
