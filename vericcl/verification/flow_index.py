from __future__ import annotations

import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Tuple

from vericcl.errors import SemanticError
from vericcl.semantics.atom import Schedule, Symbol, Transfer
from vericcl.topology.model import LaneKey


def _time(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticError("{} must be a number".format(field))
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise SemanticError("{} must be finite and non-negative".format(field))
    return result


@dataclass(frozen=True, order=True)
class LaneInterval:
    st_time: float
    ed_time: float
    transfer_id: str

    def __post_init__(self) -> None:
        start = _time(self.st_time, "lane_interval.st_time")
        end = _time(self.ed_time, "lane_interval.ed_time")
        if start > end:
            raise SemanticError("lane interval start must not exceed end")
        if not isinstance(self.transfer_id, str) or not self.transfer_id:
            raise SemanticError("lane interval transfer ID is invalid")
        object.__setattr__(self, "st_time", start)
        object.__setattr__(self, "ed_time", end)


@dataclass(frozen=True)
class LaneState:
    lane: LaneKey
    intervals: Tuple[LaneInterval, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.lane, LaneKey):
            raise SemanticError("lane state requires a LaneKey")
        intervals = tuple(sorted(self.intervals))
        if not all(isinstance(item, LaneInterval) for item in intervals):
            raise SemanticError("lane state intervals are invalid")
        object.__setattr__(self, "intervals", intervals)

    @property
    def transfer_ids(self) -> Tuple[str, ...]:
        return tuple(item.transfer_id for item in self.intervals)

    def earliest_start(
        self,
        ready_time: float,
        deadline: float,
        duration: float,
    ) -> Optional[float]:
        ready = _time(ready_time, "lane_state.ready_time")
        limit = _time(deadline, "lane_state.deadline")
        required = _time(duration, "lane_state.duration")
        if ready > limit:
            return None
        cursor = ready
        for interval in self.intervals:
            if interval.ed_time <= cursor:
                continue
            if cursor + required <= min(interval.st_time, limit):
                return cursor
            cursor = max(cursor, interval.ed_time)
            if cursor > limit:
                return None
        return cursor if cursor + required <= limit else None


@dataclass(frozen=True)
class FlowRecord:
    flow_id: str
    demand_id: str
    stage_id: int
    operator: str
    root_rank: int
    leaf_rank: int
    logical_slice_index: int
    member_slice_ids: frozenset[int]
    ranks: Tuple[int, ...]
    transfer_ids: Tuple[str, ...]
    lanes: Tuple[LaneKey, ...]
    ready_times: Tuple[float, ...]
    comparison_end: int

    @property
    def comparison_transfer_ids(self) -> Tuple[str, ...]:
        return self.transfer_ids[: self.comparison_end]


@dataclass(frozen=True)
class FlowIndex:
    flows: Tuple[FlowRecord, ...]
    lane_states: Mapping[LaneKey, LaneState]
    shared_suffix_transfer_ids: frozenset[str]

    def __post_init__(self) -> None:
        flows = tuple(self.flows)
        if len({flow.flow_id for flow in flows}) != len(flows):
            raise SemanticError("flow IDs must be unique")
        object.__setattr__(self, "flows", flows)
        object.__setattr__(
            self,
            "lane_states",
            MappingProxyType(dict(self.lane_states)),
        )
        object.__setattr__(
            self,
            "shared_suffix_transfer_ids",
            frozenset(self.shared_suffix_transfer_ids),
        )

    def flow(self, flow_id: str) -> FlowRecord:
        for flow in self.flows:
            if flow.flow_id == flow_id:
                return flow
        raise SemanticError("unknown flow ID")

    def lane(self, lane: LaneKey) -> LaneState:
        if lane not in self.lane_states:
            return LaneState(lane, ())
        return self.lane_states[lane]


def _operation_key(
    slice_id: int,
    stage_id: int,
    operator: str,
    symbol: Symbol,
) -> tuple:
    return (
        slice_id,
        stage_id,
        operator,
        symbol.src_rank,
        symbol.dst_rank,
        symbol.ready_time,
    )


def _operation_index(schedule: Schedule) -> Mapping[tuple, Transfer]:
    operations = {}
    for transfer in schedule.transfers:
        for atom in transfer.atoms:
            stage = atom.path[-1]
            key = _operation_key(
                atom.slice_id,
                stage.stage_id,
                stage.operator,
                atom.current_symbol,
            )
            previous = operations.get(key)
            if previous is not None and previous.transfer_id != transfer.transfer_id:
                raise SemanticError("path operation maps to multiple transfers")
            operations[key] = transfer
    return operations


def _maximal_stage_paths(schedule: Schedule) -> Tuple[tuple, ...]:
    paths: Dict[tuple, set] = {}
    for transfer in schedule.transfers:
        for atom in transfer.atoms:
            for stage in atom.path:
                key = (atom.slice_id, stage.stage_id, stage.operator)
                paths.setdefault(key, set()).add(tuple(stage.symbols))
    maximal = []
    for key, candidates in paths.items():
        for path in sorted(
            candidates,
            key=lambda value: (
                tuple((item.src_rank, item.dst_rank) for item in value),
                tuple(item.ready_time for item in value),
            ),
        ):
            if any(
                len(other) > len(path) and other[: len(path)] == path
                for other in candidates
            ):
                continue
            maximal.append(key + (path,))
    return tuple(
        sorted(
            maximal,
            key=lambda item: (
                item[1],
                item[2],
                item[0],
                tuple((symbol.src_rank, symbol.dst_rank) for symbol in item[3]),
            ),
        )
    )


def _common_suffix_length(left: Tuple[str, ...], right: Tuple[str, ...]) -> int:
    length = 0
    while (
        length < min(len(left), len(right))
        and left[-length - 1] == right[-length - 1]
    ):
        length += 1
    return length


def build_flow_index(schedule: Schedule) -> FlowIndex:
    if not isinstance(schedule, Schedule):
        raise SemanticError("schedule must be a Schedule")
    operations = _operation_index(schedule)
    flows = []
    for ordinal, (slice_id, stage_id, operator, symbols) in enumerate(
        _maximal_stage_paths(schedule)
    ):
        transfers = []
        for symbol in symbols:
            key = _operation_key(slice_id, stage_id, operator, symbol)
            transfer = operations.get(key)
            if transfer is None:
                raise SemanticError("flow path operation is missing from schedule")
            transfers.append(transfer)
        ranks = (symbols[0].src_rank,) + tuple(
            symbol.dst_rank for symbol in symbols
        )
        logical = slice_id % schedule.slice_count
        root = ranks[0]
        leaf = ranks[-1]
        flow_id = "flow-s{:08d}-a{:08d}-m{:08d}-r{:08d}-l{:08d}-p{:04d}".format(
            stage_id,
            logical,
            slice_id,
            root,
            leaf,
            ordinal,
        )
        demand_id = "demand-s{:08d}-{}-a{:08d}-r{:08d}-l{:08d}".format(
            stage_id,
            operator.lower(),
            logical,
            root,
            leaf,
        )
        flows.append(
            FlowRecord(
                flow_id=flow_id,
                demand_id=demand_id,
                stage_id=stage_id,
                operator=operator,
                root_rank=root,
                leaf_rank=leaf,
                logical_slice_index=logical,
                member_slice_ids=frozenset({slice_id}),
                ranks=ranks,
                transfer_ids=tuple(item.transfer_id for item in transfers),
                lanes=tuple(
                    LaneKey(item.src_rank, item.dst_rank, item.channel)
                    for item in transfers
                ),
                ready_times=tuple(symbol.ready_time for symbol in symbols),
                comparison_end=len(transfers),
            )
        )

    comparison_ends = [flow.comparison_end for flow in flows]
    shared_suffix = set()
    for left_index, left in enumerate(flows):
        for right_index in range(left_index + 1, len(flows)):
            right = flows[right_index]
            if (
                left.stage_id != right.stage_id
                or left.logical_slice_index != right.logical_slice_index
                or left.leaf_rank != right.leaf_rank
                or left.member_slice_ids == right.member_slice_ids
            ):
                continue
            length = _common_suffix_length(
                left.transfer_ids,
                right.transfer_ids,
            )
            if length == 0:
                continue
            comparison_ends[left_index] = min(
                comparison_ends[left_index],
                len(left.transfer_ids) - length,
            )
            comparison_ends[right_index] = min(
                comparison_ends[right_index],
                len(right.transfer_ids) - length,
            )
            shared_suffix.update(left.transfer_ids[-length:])
    flows = tuple(
        replace(flow, comparison_end=comparison_ends[index])
        for index, flow in enumerate(flows)
    )

    lane_intervals: Dict[LaneKey, list] = {}
    for transfer in schedule.transfers:
        lane = LaneKey(transfer.src_rank, transfer.dst_rank, transfer.channel)
        lane_intervals.setdefault(lane, []).append(
            LaneInterval(
                transfer.st_time,
                transfer.ed_time,
                transfer.transfer_id,
            )
        )
    lane_states = {
        lane: LaneState(lane, tuple(intervals))
        for lane, intervals in lane_intervals.items()
    }
    return FlowIndex(flows, lane_states, frozenset(shared_suffix))
