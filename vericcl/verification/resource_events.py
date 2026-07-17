from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Tuple

from vericcl.errors import SemanticError
from vericcl.topology.model import LinkKey, Topology


def directed_link_timeline_id(key: LinkKey) -> str:
    if not isinstance(key, LinkKey):
        raise SemanticError("key must be a LinkKey")
    return "link:{}->{}".format(key.src_rank, key.dst_rank)


def shared_resource_timeline_id(resource_id: str) -> str:
    if not isinstance(resource_id, str) or not resource_id:
        raise SemanticError("resource_id must be a non-empty string")
    return "resource:{}".format(resource_id)


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
class ResourceInterval:
    resource_id: str
    start_time_us: float
    end_time_us: float
    active_transfer_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.resource_id, str) or not self.resource_id:
            raise SemanticError("resource interval ID must be non-empty")
        start = _time(self.start_time_us, "resource_interval.start_time_us")
        end = _time(self.end_time_us, "resource_interval.end_time_us")
        if start >= end:
            raise SemanticError("resource interval must have positive duration")
        transfer_ids = tuple(sorted(set(self.active_transfer_ids)))
        if not all(
            isinstance(transfer_id, str) and transfer_id
            for transfer_id in transfer_ids
        ):
            raise SemanticError("resource interval transfer IDs are invalid")
        object.__setattr__(self, "start_time_us", start)
        object.__setattr__(self, "end_time_us", end)
        object.__setattr__(self, "active_transfer_ids", transfer_ids)

    @property
    def concurrency(self) -> int:
        return len(self.active_transfer_ids)

    @property
    def busy(self) -> bool:
        return bool(self.active_transfer_ids)


@dataclass(frozen=True)
class ResourceTimeline:
    resource_id: str
    intervals: Tuple[ResourceInterval, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.resource_id, str) or not self.resource_id:
            raise SemanticError("resource timeline ID must be non-empty")
        intervals = tuple(self.intervals)
        if not all(
            isinstance(interval, ResourceInterval)
            and interval.resource_id == self.resource_id
            for interval in intervals
        ):
            raise SemanticError("resource timeline intervals are invalid")
        for previous, current in zip(intervals, intervals[1:]):
            if not math.isclose(
                previous.end_time_us,
                current.start_time_us,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise SemanticError("resource timeline intervals are not contiguous")
        object.__setattr__(self, "intervals", intervals)

    @property
    def busy_intervals(self) -> Tuple[ResourceInterval, ...]:
        return tuple(interval for interval in self.intervals if interval.busy)

    @property
    def idle_intervals(self) -> Tuple[ResourceInterval, ...]:
        return tuple(interval for interval in self.intervals if not interval.busy)


def timeline_resource_ids(topology: Topology) -> Tuple[str, ...]:
    if not isinstance(topology, Topology):
        raise SemanticError("topology must be a Topology")
    values = [
        directed_link_timeline_id(key) for key in topology.links
    ]
    values.extend(
        shared_resource_timeline_id(resource_id)
        for resource_id in topology.shared_resources
    )
    return tuple(sorted(values))


def build_resource_timelines(
    topology: Topology,
    snapshots: Tuple[
        Tuple[float, float, Mapping[str, Tuple[str, ...]]], ...
    ],
) -> Mapping[str, ResourceTimeline]:
    resource_ids = timeline_resource_ids(topology)
    values = {resource_id: [] for resource_id in resource_ids}
    for start_time, end_time, active_by_resource in snapshots:
        for resource_id in resource_ids:
            active = tuple(active_by_resource.get(resource_id, ()))
            intervals = values[resource_id]
            if (
                intervals
                and intervals[-1].active_transfer_ids
                == tuple(sorted(set(active)))
            ):
                previous = intervals.pop()
                intervals.append(
                    ResourceInterval(
                        resource_id,
                        previous.start_time_us,
                        end_time,
                        active,
                    )
                )
            else:
                intervals.append(
                    ResourceInterval(
                        resource_id,
                        start_time,
                        end_time,
                        active,
                    )
                )
    return {
        resource_id: ResourceTimeline(resource_id, tuple(intervals))
        for resource_id, intervals in values.items()
    }
