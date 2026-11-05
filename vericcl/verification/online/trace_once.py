"""One post-warmup target invocation, with local validity and clock bounds."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping
import uuid
import math

from vericcl.errors import SemanticError
from vericcl.execution.context import StorageFailure
from vericcl.verification.online.clock_sync import align_bracketed_globaltimers, ClockOrdering
from vericcl.verification.online.trace_format import parse_partial_trace
from vericcl.xml.endpoints import EndpointType


@dataclass(frozen=True)
class CapturePlan:
    capture_id: str
    rank_ids: tuple[int, ...]
    capacity_by_rank: Mapping[int, int]
    expected_steps_by_rank: Mapping[int, int]
    xml_sha256: str
    warmup_iterations: int = 5
    timed_iterations: int = 1
    lease: object = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class RawTraceCapture:
    paths: Mapping[int, Path]
    clock_samples: tuple
    metadata_by_rank: Mapping[int, dict]
    driver_kind: str
    issues: tuple[dict, ...] = ()


@dataclass
class TraceContext:
    enabled: bool
    rank_ids: tuple[int, ...]
    driver: object = None
    capacity_by_rank: Mapping[int, int] | None = None
    lease: object = None
    run_context: object = None
    allow_test_driver: bool = False
    warmup_iterations: int = 5


@dataclass(frozen=True)
class TraceBundle:
    status: str
    global_complete: bool = False
    capture_id: str | None = None
    records_by_rank: Mapping = field(default_factory=dict)
    local_steps: Mapping = field(default_factory=dict)
    aligned_ranks: tuple[int, ...] = ()
    uncertain_relations: tuple[dict, ...] = ()
    issues: tuple[dict, ...] = ()
    global_span_us: tuple[float, float] | None = None
    complete_ranks: tuple[int, ...] = ()
    evidence_kind: str = 'none'
    alignment: object = field(default=None, repr=False, compare=False)
    device_durations_us: Mapping[int, float] = field(default_factory=dict)
    xml_sha256: str | None = None


def capture_once(sidecar, context: TraceContext) -> TraceBundle:
    if not isinstance(context.enabled, bool):
        raise SemanticError('trace_and_measure.enabled must be boolean')
    if not context.enabled:
        return TraceBundle('disabled')
    if type(context.warmup_iterations) is not int or context.warmup_iterations < 0:
        raise SemanticError('trace warmup iterations must be nonnegative')
    ranks = tuple(context.rank_ids)
    if not ranks or len(set(ranks)) != len(ranks) or any(type(rank) is not int or rank < 0 for rank in ranks):
        raise SemanticError('trace capture needs unique nonnegative ranks')
    if {entry.rank for entry in sidecar.entries.values()} - set(ranks):
        raise SemanticError('trace rank set does not cover the schedule')
    expected = {rank: sum(entry.rank == rank for entry in sidecar.entries.values()) for rank in ranks}
    capacity = {rank: max(1, count) for rank, count in expected.items()} if context.capacity_by_rank is None else dict(context.capacity_by_rank)
    if set(capacity) != set(ranks) or any(type(capacity[rank]) is not int or capacity[rank] < max(1, count) for rank, count in expected.items()):
        return TraceBundle('capacity_insufficient', issues=({'rank': None, 'reason': 'trace_capacity_below_single_invocation'},),
                           xml_sha256=sidecar.xml_sha256)
    plan = CapturePlan(uuid.uuid4().hex, ranks, capacity, expected, sidecar.xml_sha256, warmup_iterations=context.warmup_iterations, lease=context.lease)
    if context.run_context is not None:
        context.run_context.write_json('jobs/trace-' + plan.capture_id + '/plan.json',
            {key: getattr(plan, key) for key in ('capture_id', 'rank_ids', 'capacity_by_rank', 'expected_steps_by_rank', 'xml_sha256', 'warmup_iterations', 'timed_iterations')})
    try:
        if context.lease is not None:
            context.lease.check()
        raw = context.driver.capture_once(plan)
        if not isinstance(raw, RawTraceCapture) or (raw.driver_kind != 'native_msccl' and not (context.allow_test_driver and raw.driver_kind == 'unit_fixture')):
            raise SemanticError('trace capture requires the native once-only driver')
    except StorageFailure:
        raise
    except Exception as error:
        return TraceBundle('capture_failed', capture_id=plan.capture_id,
            issues=({'rank': None, 'reason': type(error).__name__ + ': ' + str(error)},), xml_sha256=sidecar.xml_sha256)
    records, local, complete, issues, device_durations = {}, {}, [], list(raw.issues), {}
    for rank in ranks:
        metadata = raw.metadata_by_rank.get(rank, {})
        event_time = metadata.get('target_event_time_us')
        if (isinstance(event_time, bool) or not isinstance(event_time, (int, float)) or not math.isfinite(event_time) or event_time <= 0):
            issues.append(dict(rank=rank, reason='target_device_duration_missing_or_invalid'))
        else:
            device_durations[rank] = float(event_time)
        if (metadata.get('warmup_iterations') != plan.warmup_iterations or metadata.get('reset_after_warmup') is not True
                or metadata.get('executed_after_reset') != 1 or metadata.get('gpu_completed') is not True
                or metadata.get('correct') is not True or metadata.get('clock_domain') != 'globaltimer_ns'):
            issues.append(dict(rank=rank, reason='missing_or_invalid_capture_receipt'))
            continue
        if rank not in raw.paths:
            issues.append(dict(rank=rank, reason='missing_rank_file'))
            continue
        try:
            partial = parse_partial_trace(raw.paths[rank], sidecar, expected_rank=rank)
            if partial.capacity < capacity[rank]:
                raise SemanticError('native trace capacity differs from capture plan')
            iterations = {record.iteration for record in partial.records}
            if len(iterations) > 1 or (expected[rank] and len(iterations) != 1):
                raise SemanticError('trace must contain exactly one target invocation')
        except (OSError, SemanticError) as error:
            issues.append(dict(rank=rank, reason=str(error)))
            continue
        issues.extend(partial.issues)
        records[rank] = partial.records
        if not partial.issues:
            complete.append(rank)
        origin = min((record.raw.tb_reach for record in partial.records), default=0)
        local[rank] = tuple(dict(transfer_id=record.transfer_id, tb_id=record.tb_id, step_index=record.step_index,
            channel=record.raw.channel, iteration=record.iteration,
            start_from_first_reach_us=(record.raw.transfer_start - origin) / 1000,
            end_from_first_reach_us=(record.raw.transfer_end - origin) / 1000,
            dependency_wait_us=(record.raw.dependency_done - record.raw.tb_reach) / 1000,
            transfer_us=(record.raw.transfer_end - record.raw.transfer_start) / 1000,
            payload_bytes=record.metadata.payload_bytes, atom_ids=record.atom_ids, flow_ids=record.flow_ids) for record in partial.records)
    alignment, clock_issues = align_bracketed_globaltimers(records, raw.clock_samples)
    issues.extend(clock_issues)
    aligned = () if alignment is None else tuple(sorted(alignment.transforms))
    uncertain = []
    by_transfer = {}
    for values in records.values():
        for record in values:
            by_transfer.setdefault(record.transfer_id, []).append(record)
    # Pair physical endpoints only with the same invocation and a qualified clock map.
    for transfer_id, ends in by_transfer.items():
        network = any(record.endpoint_type is EndpointType.SEND for record in ends)
        if network and (len(ends) != 2 or len({record.iteration for record in ends}) != 1 or any(record.rank not in aligned for record in ends)):
            uncertain.append(dict(transfer_id=transfer_id, reason='missing_endpoint_or_clock'))
            continue
        if network:
            for name in ('transfer_start', 'transfer_end'):
                if alignment.compare(ends[0].rank, getattr(ends[0].raw, name), ends[1].rank, getattr(ends[1].raw, name)) is ClockOrdering.UNORDERED:
                    uncertain.append(dict(transfer_id=transfer_id, field=name, reason='endpoint_order_within_clock_error'))
    # Dependency relations are reported independently, without erasing valid local intervals.
    for transfer_id, ends in by_transfer.items():
        for record in ends:
            for predecessor in record.semantic_predecessor_ids:
                previous = by_transfer.get(predecessor, ())
                if not previous:
                    uncertain.append(dict(transfer_id=transfer_id, predecessor=predecessor, reason='missing_dependency_endpoint'))
                for prior in previous:
                    if prior.rank == record.rank:
                        continue
                    if prior.rank not in aligned or record.rank not in aligned or alignment.compare(prior.rank, prior.raw.transfer_end, record.rank, record.raw.tb_reach) is ClockOrdering.UNORDERED:
                        uncertain.append(dict(transfer_id=transfer_id, predecessor=predecessor, reason='cross_rank_dependency_uncertain'))
    invocation_ids = {record.iteration for values in records.values() for record in values}
    if len(invocation_ids) > 1:
        issues.append(dict(rank=None, reason='target_invocation_identity_differs_across_ranks'))
    full = (not any(issue.get('blocks_acceptance', False) for issue in issues)
            and set(device_durations) == set(ranks) and len(invocation_ids) <= 1
            and set(complete) == set(ranks) and all(not records[rank] or rank in aligned for rank in ranks))
    span = None
    if full:
        starts = [alignment.timestamp(rank, record.raw.tb_reach) for rank, values in records.items() for record in values]
        ends = [alignment.timestamp(rank, record.raw.transfer_end) for rank, values in records.items() for record in values]
        if starts:
            span = (max(0, max(value.value_us - value.uncertainty_us for value in ends) - min(value.value_us + value.uncertainty_us for value in starts)),
                    max(value.value_us + value.uncertainty_us for value in ends) - min(value.value_us - value.uncertainty_us for value in starts))
    result = TraceBundle('complete' if full else 'partial', full, plan.capture_id, records, local, aligned,
                         tuple(uncertain), tuple(issues), span, tuple(complete), raw.driver_kind, alignment, device_durations,
                         sidecar.xml_sha256)
    if context.run_context is not None:
        context.run_context.write_json('reports/trace-' + plan.capture_id + '.json', dict(status=result.status,
            global_complete=full, evidence_kind=raw.driver_kind, local_steps=local, aligned_ranks=aligned,
            uncertain_relations=uncertain, issues=issues, global_span_us=span, device_durations_us=device_durations))
    return result
