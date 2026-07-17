from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Dict, Iterable, Optional, Tuple

from vericcl.errors import SemanticError
from vericcl.topology.model import LaneKey
from vericcl.verification.online.clock_sync import (
    AlignedTimestamp,
    ClockAlignment,
    ClockOrdering,
)
from vericcl.verification.online.trace_format import StepTraceRecord
from vericcl.xml.endpoints import EndpointType
from vericcl.xml.trace_sidecar import TraceSidecar


class WaitClass(str, Enum):
    HEAD_OF_LINE = "head_of_line_wait"
    DEPENDENCY = "dependency_wait"
    PEER_RESOURCE = "peer_resource_wait"
    TRANSFER = "transfer_duration"


@dataclass(frozen=True)
class WaitDurations:
    head_of_line_wait_us: float
    dependency_wait_us: float
    peer_resource_wait_us: float
    transfer_duration_us: float


@dataclass(frozen=True)
class PhysicalTransferInterval:
    transfer_id: str
    iteration: int
    send: Optional[StepTraceRecord]
    receive: Optional[StepTraceRecord]
    local: Optional[StepTraceRecord]
    physical_start: AlignedTimestamp
    physical_end: AlignedTimestamp
    endpoint_order_uncertain: bool

    @property
    def physical_start_us(self) -> float:
        return self.physical_start.value_us

    @property
    def physical_end_us(self) -> float:
        return self.physical_end.value_us


@dataclass(frozen=True)
class StepWaitAnalysis:
    record: StepTraceRecord
    semantic_ready: AlignedTimestamp
    waits: WaitDurations
    ordering_confident: bool

    @property
    def transfer_id(self) -> str:
        return self.record.transfer_id

    @property
    def semantic_ready_us(self) -> float:
        return self.semantic_ready.value_us


@dataclass(frozen=True)
class BottleneckRecord:
    transfer_id: str
    stage_id: int
    endpoint_type: EndpointType
    atom_ids: Tuple[str, ...]
    flow_ids: Tuple[str, ...]
    rank: int
    tb_id: int
    step_index: int
    iteration: int
    lane: Optional[LaneKey]
    wait_class: WaitClass
    duration_us: float
    ordering_confident: bool


@dataclass(frozen=True)
class TraceAnalysis:
    intervals: Tuple[PhysicalTransferInterval, ...]
    step_waits: Tuple[StepWaitAnalysis, ...]
    bottlenecks: Tuple[BottleneckRecord, ...]
    uncertain_comparisons: Tuple[str, ...]
    tuning_eligible: bool


def _aligned_max(
    left: AlignedTimestamp,
    right: AlignedTimestamp,
) -> AlignedTimestamp:
    selected = left if left.value_us >= right.value_us else right
    return AlignedTimestamp(
        selected.value_us,
        max(left.uncertainty_us, right.uncertainty_us),
    )


def _record_timestamp(
    record: StepTraceRecord,
    field: str,
    alignment: ClockAlignment,
) -> AlignedTimestamp:
    return alignment.timestamp(record.rank, getattr(record.raw, field))


def pair_endpoints(
    send: StepTraceRecord,
    receive: StepTraceRecord,
    alignment: ClockAlignment,
) -> PhysicalTransferInterval:
    if not isinstance(send, StepTraceRecord) or not isinstance(
        receive,
        StepTraceRecord,
    ):
        raise SemanticError("endpoint pairing requires step trace records")
    if not isinstance(alignment, ClockAlignment):
        raise SemanticError("endpoint pairing requires clock alignment")
    if send.endpoint_type is not EndpointType.SEND or receive.endpoint_type not in {
        EndpointType.RECV,
        EndpointType.RECV_REDUCE_COPY,
    }:
        raise SemanticError("physical transfer endpoint types are invalid")
    if (
        send.transfer_id != receive.transfer_id
        or send.iteration != receive.iteration
        or send.rank != receive.metadata.peer
        or receive.rank != send.metadata.peer
        or send.lane != receive.lane
        or send.atom_ids != receive.atom_ids
        or send.flow_ids != receive.flow_ids
        or send.semantic_predecessor_ids
        != receive.semantic_predecessor_ids
    ):
        raise SemanticError("physical transfer endpoint metadata differs")

    send_start = _record_timestamp(send, "transfer_start", alignment)
    receive_start = _record_timestamp(receive, "transfer_start", alignment)
    send_end = _record_timestamp(send, "transfer_end", alignment)
    receive_end = _record_timestamp(receive, "transfer_end", alignment)
    start_order = alignment.compare_timestamps(send_start, receive_start)
    end_order = alignment.compare_timestamps(send_end, receive_end)
    uncertainty = (
        start_order is ClockOrdering.UNORDERED
        and send_start.uncertainty_us + receive_start.uncertainty_us > 0.0
    ) or (
        end_order is ClockOrdering.UNORDERED
        and send_end.uncertainty_us + receive_end.uncertainty_us > 0.0
    )
    return PhysicalTransferInterval(
        transfer_id=send.transfer_id,
        iteration=send.iteration,
        send=send,
        receive=receive,
        local=None,
        physical_start=_aligned_max(send_start, receive_start),
        physical_end=_aligned_max(send_end, receive_end),
        endpoint_order_uncertain=uncertainty,
    )


def _local_interval(
    record: StepTraceRecord,
    alignment: ClockAlignment,
) -> PhysicalTransferInterval:
    if record.endpoint_type is not EndpointType.COPY:
        raise SemanticError("local interval requires a copy record")
    return PhysicalTransferInterval(
        transfer_id=record.transfer_id,
        iteration=record.iteration,
        send=None,
        receive=None,
        local=record,
        physical_start=_record_timestamp(record, "transfer_start", alignment),
        physical_end=_record_timestamp(record, "transfer_end", alignment),
        endpoint_order_uncertain=False,
    )


def decompose_waits(
    *,
    tb_reach_us: float,
    dependency_done_us: float,
    physical_start_us: float,
    physical_end_us: float,
    semantic_ready_us: float,
) -> WaitDurations:
    values = (
        tb_reach_us,
        dependency_done_us,
        physical_start_us,
        physical_end_us,
        semantic_ready_us,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        raise SemanticError("wait decomposition values must be finite numbers")
    if physical_end_us < physical_start_us:
        raise SemanticError("physical transfer interval is reversed")
    return WaitDurations(
        head_of_line_wait_us=max(0.0, tb_reach_us - semantic_ready_us),
        dependency_wait_us=max(
            0.0,
            dependency_done_us - tb_reach_us,
        ),
        peer_resource_wait_us=max(
            0.0,
            physical_start_us - dependency_done_us,
        ),
        transfer_duration_us=physical_end_us - physical_start_us,
    )


def _build_intervals(
    records: Tuple[StepTraceRecord, ...],
    alignment: ClockAlignment,
) -> Tuple[PhysicalTransferInterval, ...]:
    grouped: Dict[Tuple[str, int], list] = {}
    for record in records:
        grouped.setdefault((record.transfer_id, record.iteration), []).append(
            record
        )
    intervals = []
    for key in sorted(grouped):
        group = tuple(grouped[key])
        copies = tuple(
            record
            for record in group
            if record.endpoint_type is EndpointType.COPY
        )
        if copies:
            if len(group) != 1:
                raise SemanticError("local copy trace has extra endpoints")
            intervals.append(_local_interval(copies[0], alignment))
            continue
        sends = tuple(
            record
            for record in group
            if record.endpoint_type is EndpointType.SEND
        )
        receives = tuple(
            record
            for record in group
            if record.endpoint_type
            in {EndpointType.RECV, EndpointType.RECV_REDUCE_COPY}
        )
        if len(sends) != 1 or len(receives) != 1 or len(group) != 2:
            raise SemanticError(
                "communication trace must contain both endpoints exactly once"
            )
        intervals.append(pair_endpoints(sends[0], receives[0], alignment))
    return tuple(intervals)


def _semantic_ready(
    record: StepTraceRecord,
    intervals: Dict[Tuple[str, int], PhysicalTransferInterval],
    tb_reach: AlignedTimestamp,
) -> Tuple[AlignedTimestamp, bool]:
    predecessors = record.semantic_predecessor_ids
    if not predecessors:
        return tb_reach, False
    values = []
    for predecessor in predecessors:
        key = (predecessor, record.iteration)
        if key not in intervals:
            raise SemanticError("semantic predecessor trace is missing")
        values.append(intervals[key].physical_end)
    selected = max(values, key=lambda value: value.value_us)
    uncertain = False
    for value in values:
        if value is selected:
            continue
        if (
            ClockAlignment.compare_timestamps(value, selected)
            is ClockOrdering.UNORDERED
            and value.uncertainty_us + selected.uncertainty_us > 0.0
        ):
            uncertain = True
    return selected, uncertain


def _bottlenecks(wait: StepWaitAnalysis) -> Iterable[BottleneckRecord]:
    if wait.record.endpoint_type is EndpointType.COPY:
        return
    values = (
        (WaitClass.HEAD_OF_LINE, wait.waits.head_of_line_wait_us),
        (WaitClass.DEPENDENCY, wait.waits.dependency_wait_us),
        (WaitClass.PEER_RESOURCE, wait.waits.peer_resource_wait_us),
        (WaitClass.TRANSFER, wait.waits.transfer_duration_us),
    )
    for wait_class, duration in values:
        if (
            wait_class is WaitClass.TRANSFER
            and wait.record.endpoint_type is not EndpointType.SEND
        ):
            continue
        if duration <= 0.0:
            continue
        record = wait.record
        yield BottleneckRecord(
            transfer_id=record.transfer_id,
            stage_id=record.metadata.stage_id,
            endpoint_type=record.endpoint_type,
            atom_ids=record.atom_ids,
            flow_ids=record.flow_ids,
            rank=record.rank,
            tb_id=record.tb_id,
            step_index=record.step_index,
            iteration=record.iteration,
            lane=record.lane,
            wait_class=wait_class,
            duration_us=duration,
            ordering_confident=wait.ordering_confident,
        )


def analyze_trace(
    records: Iterable[StepTraceRecord],
    sidecar_or_alignment,
    alignment: Optional[ClockAlignment] = None,
) -> TraceAnalysis:
    if alignment is None:
        sidecar = None
        alignment = sidecar_or_alignment
    else:
        sidecar = sidecar_or_alignment
    if not isinstance(alignment, ClockAlignment):
        raise SemanticError("trace analysis requires clock alignment")
    if sidecar is not None and not isinstance(sidecar, TraceSidecar):
        raise SemanticError("trace analysis sidecar is invalid")
    try:
        normalized = tuple(records)
    except TypeError as error:
        raise SemanticError("trace records must be iterable") from error
    if not normalized or not all(
        isinstance(record, StepTraceRecord) for record in normalized
    ):
        raise SemanticError("trace analysis requires step records")
    if sidecar is not None:
        for record in normalized:
            if sidecar.entry(
                record.rank,
                record.tb_id,
                record.step_index,
            ) != record.metadata:
                raise SemanticError("trace record metadata differs from sidecar")

    interval_values = _build_intervals(normalized, alignment)
    intervals = {
        (interval.transfer_id, interval.iteration): interval
        for interval in interval_values
    }
    uncertain = []
    for interval in interval_values:
        if interval.endpoint_order_uncertain:
            uncertain.append(
                "{}:{}:endpoint_order".format(
                    interval.transfer_id,
                    interval.iteration,
                )
            )

    step_waits = []
    for record in sorted(
        normalized,
        key=lambda item: (
            item.iteration,
            item.rank,
            item.tb_id,
            item.step_index,
            item.transfer_id,
        ),
    ):
        interval = intervals[(record.transfer_id, record.iteration)]
        tb_reach = _record_timestamp(record, "tb_reach", alignment)
        dependency_done = _record_timestamp(
            record,
            "dependency_done",
            alignment,
        )
        semantic_ready, predecessor_max_uncertain = _semantic_ready(
            record,
            intervals,
            tb_reach,
        )
        head_order = alignment.compare_timestamps(
            semantic_ready,
            tb_reach,
        )
        peer_order = alignment.compare_timestamps(
            dependency_done,
            interval.physical_start,
        )
        head_uncertain = (
            bool(record.semantic_predecessor_ids)
            and head_order is ClockOrdering.UNORDERED
            and semantic_ready.uncertainty_us + tb_reach.uncertainty_us > 0.0
        )
        peer_uncertain = (
            peer_order is ClockOrdering.UNORDERED
            and dependency_done.uncertainty_us
            + interval.physical_start.uncertainty_us
            > 0.0
        )
        confident = not (
            predecessor_max_uncertain
            or head_uncertain
            or peer_uncertain
            or interval.endpoint_order_uncertain
        )
        if not confident:
            uncertain.append(
                "{}:{}:r{}:tb{}:s{}".format(
                    record.transfer_id,
                    record.iteration,
                    record.rank,
                    record.tb_id,
                    record.step_index,
                )
            )
        waits = decompose_waits(
            tb_reach_us=tb_reach.value_us,
            dependency_done_us=dependency_done.value_us,
            physical_start_us=interval.physical_start_us,
            physical_end_us=interval.physical_end_us,
            semantic_ready_us=semantic_ready.value_us,
        )
        step_waits.append(
            StepWaitAnalysis(record, semantic_ready, waits, confident)
        )
    bottlenecks = tuple(
        bottleneck
        for wait in step_waits
        for bottleneck in _bottlenecks(wait)
    )
    uncertain_values = tuple(sorted(set(uncertain)))
    return TraceAnalysis(
        intervals=interval_values,
        step_waits=tuple(step_waits),
        bottlenecks=bottlenecks,
        uncertain_comparisons=uncertain_values,
        tuning_eligible=not uncertain_values,
    )
