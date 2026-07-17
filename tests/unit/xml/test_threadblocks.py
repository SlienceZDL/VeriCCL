from dataclasses import replace

import pytest

from vericcl.errors import SemanticError
from vericcl.semantics.collective import CollectiveKind
from vericcl.xml.buffers import build_buffer_plan
from vericcl.xml.dependencies import build_transfer_dag
from vericcl.xml.endpoints import EndpointType, lower_endpoints
from vericcl.xml.list_scheduler import schedule_threadblocks
from vericcl.xml.threadblocks import Threadblock, ThreadblockKey

from tests.unit.xml.helpers import (
    inplace_alltoall_overwrite_schedule,
    resolved,
    two_send_same_lane_schedule,
)


pytestmark = pytest.mark.phase04


def _lower(schedule, inputs):
    buffers = build_buffer_plan(schedule, inputs)
    endpoints = lower_endpoints(schedule, buffers)
    dag = build_transfer_dag(endpoints, schedule, buffers)
    return schedule_threadblocks(endpoints, dag)


def test_send_and_receive_use_different_unidirectional_threadblocks():
    program = _lower(
        inplace_alltoall_overwrite_schedule(),
        resolved(CollectiveKind.ALL_TO_ALL, inplace=True),
    )

    for threadblock in program.threadblocks:
        types = {
            step.xml_type
            for step in threadblock.steps
            if step.xml_type is not EndpointType.NOP
        }
        assert not (
            EndpointType.SEND in types
            and types.intersection(
                {EndpointType.RECV, EndpointType.RECV_REDUCE_COPY}
            )
        )


def test_paired_endpoints_have_matching_lane_order():
    program = _lower(
        two_send_same_lane_schedule(),
        resolved(CollectiveKind.BROADCAST, ranks=2, slices=2),
    )

    send = next(
        tb
        for tb in program.threadblocks
        if tb.key.rank == 0 and tb.key.direction == "send" and tb.key.peer == 1
    )
    recv = next(
        tb
        for tb in program.threadblocks
        if tb.key.rank == 1 and tb.key.direction == "recv" and tb.key.peer == 0
    )
    assert [step.transfer_id for step in send.steps] == [
        "lane-send-0",
        "lane-send-1",
    ]
    assert [step.transfer_id for step in recv.steps] == [
        "lane-send-0",
        "lane-send-1",
    ]
    assert program.inversion_count == 0


def test_local_copies_use_dedicated_local_threadblocks():
    program = _lower(
        inplace_alltoall_overwrite_schedule(),
        resolved(CollectiveKind.ALL_TO_ALL, inplace=True),
    )

    local = [tb for tb in program.threadblocks if tb.key.direction == "copy"]
    assert local
    assert all(
        tb.send_peer == -1
        and tb.recv_peer == -1
        and tb.key.channel == -1
        and all(step.xml_type is EndpointType.COPY for step in tb.steps)
        for tb in local
    )


@pytest.mark.parametrize(
    "args,match",
    [
        ((True, "send", 1, 0), "rank"),
        ((0, "both", 1, 0), "direction"),
        ((0, "copy", 1, 0), "sentinels"),
        ((0, "send", 0, 0), "key is invalid"),
    ],
)
def test_threadblock_key_rejects_invalid_lane_identity(args, match):
    with pytest.raises(SemanticError, match=match):
        ThreadblockKey(*args)


def test_threadblock_rejects_direction_and_lane_mismatches():
    program = _lower(
        two_send_same_lane_schedule(),
        resolved(CollectiveKind.BROADCAST, ranks=2, slices=2),
    )
    send = next(tb for tb in program.threadblocks if tb.key.direction == "send")

    with pytest.raises(SemanticError, match="direction"):
        Threadblock(
            key=replace(send.key, direction="recv"),
            tb_id=send.tb_id,
            steps=send.steps,
        )
    with pytest.raises(SemanticError, match="lane"):
        Threadblock(
            key=replace(send.key, channel=send.key.channel + 1),
            tb_id=send.tb_id,
            steps=send.steps,
        )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("step_id", "", "non-empty"),
        ("xml_type", "s", "EndpointType"),
        ("has_dependence", 1, "boolean"),
        ("endpoint_id", None, "endpoint ID"),
    ],
)
def test_xml_step_rejects_invalid_core_fields(field, value, match):
    program = _lower(
        two_send_same_lane_schedule(),
        resolved(CollectiveKind.BROADCAST, ranks=2, slices=2),
    )
    step = next(
        step
        for step in program.steps_by_id.values()
        if step.xml_type is EndpointType.SEND
    )

    with pytest.raises(SemanticError, match=match):
        replace(step, **{field: value})


def test_program_rejects_invalid_indexes_and_unknown_lookup():
    program = _lower(
        two_send_same_lane_schedule(),
        resolved(CollectiveKind.BROADCAST, ranks=2, slices=2),
    )

    with pytest.raises(SemanticError, match="steps_by_id"):
        replace(program, steps_by_id={})
    with pytest.raises(SemanticError, match="two steps"):
        replace(program, transfer_steps={"lane-send-0": ("lane-send-0:s",)})
    with pytest.raises(SemanticError, match="node step"):
        replace(program, node_steps={"missing": ("missing",)})
    with pytest.raises(SemanticError, match="referenced step"):
        replace(program, referenced_step_ids={"missing"})
    with pytest.raises(SemanticError, match="inversion_count"):
        replace(program, inversion_count=-1)
    with pytest.raises(SemanticError, match="unknown step"):
        program.threadblock_for_step("missing")
