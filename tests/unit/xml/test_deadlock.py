from dataclasses import replace

import pytest

from vericcl.semantics.collective import CollectiveKind
from vericcl.xml.buffers import build_buffer_plan
from vericcl.xml.deadlock import simulate_endpoint_execution
from vericcl.xml.dependencies import build_transfer_dag
from vericcl.xml.endpoints import lower_endpoints
from vericcl.xml.list_scheduler import schedule_threadblocks

from tests.unit.xml.helpers import resolved, two_send_same_lane_schedule


pytestmark = pytest.mark.phase04


def _program():
    schedule = two_send_same_lane_schedule()
    buffers = build_buffer_plan(
        schedule,
        resolved(CollectiveKind.BROADCAST, ranks=2, slices=2),
    )
    endpoints = lower_endpoints(schedule, buffers)
    return schedule_threadblocks(
        endpoints,
        build_transfer_dag(endpoints, schedule, buffers),
    )


def test_scheduled_program_completes_without_deadlock():
    result = simulate_endpoint_execution(_program())

    assert result.deadlocked is False
    assert result.blocked_transfer_ids == frozenset()


def test_crossed_send_receive_heads_report_deadlock():
    program = _program()
    recv_tb = next(
        tb
        for tb in program.threadblocks
        if tb.key.rank == 1 and tb.key.direction == "recv" and tb.key.peer == 0
    )
    crossed_tb = replace(recv_tb, steps=tuple(reversed(recv_tb.steps)))
    crossed = replace(
        program,
        threadblocks=tuple(
            crossed_tb if tb == recv_tb else tb for tb in program.threadblocks
        ),
    )

    result = simulate_endpoint_execution(crossed)

    assert result.deadlocked is True
    assert result.blocked_transfer_ids == frozenset(
        {"lane-send-0", "lane-send-1"}
    )
    assert result.tb_heads[(0, 1)] == "lane-send-0:s"
    assert result.tb_heads[(1, 0)] == "lane-send-1:recv"
