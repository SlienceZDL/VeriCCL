from types import MappingProxyType

import pytest

from vericcl.errors import SemanticError
from vericcl.topology.model import LaneKey
from vericcl.verification.online.clock_sync import (
    ClockAlignment,
    ClockTransform,
)
from vericcl.verification.online.trace_analysis import (
    WaitClass,
    analyze_trace,
    decompose_waits,
    pair_endpoints,
)
from vericcl.verification.online.trace_format import (
    RawStepTraceRecord,
    StepTraceRecord,
)
from vericcl.xml.endpoints import EndpointType
from vericcl.xml.trace_sidecar import TraceSidecar, TraceStepMetadata


pytestmark = pytest.mark.phase06


def _alignment(uncertainty=0.0):
    transform = ClockTransform(0, 1.0, 0.0, uncertainty, 3)
    return ClockAlignment(MappingProxyType({0: transform, 1: replace_rank(transform, 1)}))


def replace_rank(transform, rank):
    return ClockTransform(
        rank,
        transform.slope_us_per_tick,
        transform.intercept_us,
        transform.uncertainty_us,
        transform.sample_count,
    )


def _record(
    transfer_id,
    endpoint_type,
    rank,
    peer,
    *,
    tb_reach,
    dependency_done,
    start,
    end,
    predecessors=(),
    step_index=0,
):
    runtime_type = {
        EndpointType.SEND: 0,
        EndpointType.RECV: 1,
        EndpointType.RECV_REDUCE_COPY: 4,
    }[endpoint_type]
    lane = LaneKey(rank, peer, 0) if endpoint_type is EndpointType.SEND else LaneKey(peer, rank, 0)
    metadata = TraceStepMetadata(
        rank=rank,
        tb_id=rank,
        step_index=step_index,
        xml_step_index=step_index,
        step_id="{}:{}".format(transfer_id, endpoint_type.value),
        transfer_id=transfer_id,
        endpoint_type=endpoint_type,
        runtime_endpoint_type=runtime_type,
        peer=peer,
        runtime_channel=0,
        stage_id=0,
        atom_ids=("{}:atom-s00000000".format(transfer_id),),
        flow_ids=("flow-{}".format(transfer_id),),
        lane=lane,
        semantic_predecessor_ids=tuple(predecessors),
        member_slice_ids=frozenset({0}),
    )
    raw = RawStepTraceRecord(
        rank=rank,
        tb_id=rank,
        step_index=step_index,
        endpoint_type=runtime_type,
        peer=peer,
        channel=0,
        iteration=0,
        tb_reach=tb_reach,
        dependency_done=dependency_done,
        transfer_start=start,
        transfer_end=end,
        flags=0,
        reserved=0,
    )
    return StepTraceRecord(raw, metadata)


def _pair(transfer_id, *, start_send, end_send, start_recv, end_recv, **values):
    common = {
        "tb_reach": values.get("tb_reach", min(start_send, start_recv)),
        "dependency_done": values.get(
            "dependency_done", min(start_send, start_recv)
        ),
        "predecessors": values.get("predecessors", ()),
    }
    return (
        _record(
            transfer_id,
            EndpointType.SEND,
            0,
            1,
            start=start_send,
            end=end_send,
            **common,
        ),
        _record(
            transfer_id,
            EndpointType.RECV,
            1,
            0,
            start=start_recv,
            end=end_recv,
            **common,
        ),
    )


def _copy_record():
    metadata = TraceStepMetadata(
        rank=0,
        tb_id=2,
        step_index=0,
        xml_step_index=0,
        step_id="copy:cpy",
        transfer_id="copy",
        endpoint_type=EndpointType.COPY,
        runtime_endpoint_type=6,
        peer=-1,
        runtime_channel=0,
        stage_id=-1,
        atom_ids=(),
        flow_ids=(),
        lane=None,
        semantic_predecessor_ids=(),
        member_slice_ids=frozenset(),
    )
    raw = RawStepTraceRecord(
        rank=0,
        tb_id=2,
        step_index=0,
        endpoint_type=6,
        peer=-1,
        channel=0,
        iteration=0,
        tb_reach=1,
        dependency_done=1,
        transfer_start=2,
        transfer_end=3,
        flags=0,
        reserved=0,
    )
    return StepTraceRecord(raw, metadata)


def test_physical_interval_uses_both_endpoints():
    send, recv = _pair(
        "x",
        start_send=10,
        end_send=20,
        start_recv=12,
        end_recv=25,
    )

    pair = pair_endpoints(send, recv, _alignment())

    assert pair.physical_start_us == 12
    assert pair.physical_end_us == 25


def test_pair_retains_sender_local_interval():
    send, recv = _pair(
        "x",
        start_send=10,
        end_send=20,
        start_recv=100,
        end_recv=110,
    )

    interval = pair_endpoints(send, recv, _alignment(uncertainty=50.0))

    assert interval.endpoint_order_uncertain is True
    assert interval.sender_start_us == 10.0
    assert interval.sender_end_us == 20.0


def test_analysis_accepts_and_validates_explicit_sidecar():
    records = _pair(
        "x",
        start_send=10,
        end_send=20,
        start_recv=12,
        end_recv=25,
    )
    sidecar = TraceSidecar(
        xml_sha256="a" * 64,
        schedule_id="test",
        rank_count=2,
        entries={record.metadata.key: record.metadata for record in records},
    )

    analysis = analyze_trace(records, sidecar, _alignment())

    assert len(analysis.intervals) == 1


def test_wait_decomposition_matches_semantic_formulas():
    waits = decompose_waits(
        tb_reach_us=15,
        dependency_done_us=19,
        physical_start_us=24,
        physical_end_us=34,
        semantic_ready_us=10,
    )

    assert waits.head_of_line_wait_us == 5
    assert waits.dependency_wait_us == 4
    assert waits.peer_resource_wait_us == 5
    assert waits.transfer_duration_us == 10


def test_analysis_uses_maximum_end_of_every_semantic_predecessor():
    predecessor_a = _pair(
        "a", start_send=1, end_send=8, start_recv=1, end_recv=8
    )
    predecessor_b = _pair(
        "b", start_send=2, end_send=9, start_recv=2, end_recv=10
    )
    current = _pair(
        "c",
        start_send=24,
        end_send=34,
        start_recv=24,
        end_recv=34,
        tb_reach=15,
        dependency_done=19,
        predecessors=("a", "b"),
    )

    analysis = analyze_trace(
        predecessor_a + predecessor_b + current,
        _alignment(),
    )
    current_waits = tuple(
        wait for wait in analysis.step_waits if wait.transfer_id == "c"
    )

    assert len(current_waits) == 2
    assert {wait.semantic_ready_us for wait in current_waits} == {10}
    assert all(wait.waits.head_of_line_wait_us == 5 for wait in current_waits)
    assert all(wait.waits.dependency_wait_us == 4 for wait in current_waits)
    assert all(wait.waits.peer_resource_wait_us == 5 for wait in current_waits)
    assert analysis.tuning_eligible is True
    assert {
        bottleneck.wait_class for bottleneck in analysis.bottlenecks
    } >= {
        WaitClass.HEAD_OF_LINE,
        WaitClass.DEPENDENCY,
        WaitClass.PEER_RESOURCE,
    }
    assert all(
        bottleneck.transfer_id
        and bottleneck.atom_ids
        and bottleneck.flow_ids
        and bottleneck.lane is not None
        for bottleneck in analysis.bottlenecks
    )


def test_analysis_requires_both_communication_endpoints():
    send, _ = _pair(
        "x", start_send=1, end_send=2, start_recv=1, end_recv=2
    )
    with pytest.raises(SemanticError, match="both endpoints"):
        analyze_trace((send,), _alignment())


def test_local_copy_participates_in_readiness_without_flow_bottlenecks():
    analysis = analyze_trace((_copy_record(),), _alignment())

    assert len(analysis.intervals) == 1
    assert analysis.intervals[0].local is not None
    assert analysis.step_waits[0].waits.transfer_duration_us == 1
    assert analysis.bottlenecks == ()


def test_uncertain_ordering_blocks_online_tuning():
    predecessor = _pair(
        "a", start_send=1, end_send=10, start_recv=1, end_recv=10
    )
    current = _pair(
        "b",
        start_send=11,
        end_send=20,
        start_recv=11,
        end_recv=20,
        tb_reach=10,
        dependency_done=10,
        predecessors=("a",),
    )

    analysis = analyze_trace(predecessor + current, _alignment(0.2))

    assert analysis.tuning_eligible is False
    assert analysis.uncertain_comparisons
