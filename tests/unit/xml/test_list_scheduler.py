import pytest

from vericcl.semantics.collective import CollectiveKind
from vericcl.xml.buffers import build_buffer_plan
from vericcl.xml.dependencies import build_transfer_dag
from vericcl.xml.endpoints import EndpointType, lower_endpoints
from vericcl.xml.list_scheduler import schedule_threadblocks

from tests.unit.xml.helpers import allreduce_star_schedule, resolved


pytestmark = pytest.mark.phase04


def test_three_contributor_join_uses_two_nops_and_latest_direct_dependency():
    schedule = allreduce_star_schedule()
    buffers = build_buffer_plan(
        schedule,
        resolved(CollectiveKind.ALL_REDUCE, ranks=4, slices=1),
    )
    endpoints = lower_endpoints(schedule, buffers)
    dag = build_transfer_dag(endpoints, schedule, buffers)

    program = schedule_threadblocks(endpoints, dag)

    for destination in range(1, 4):
        transfer_id = "allreduce-send-{}".format(destination)
        send_step_id = next(
            step_id
            for step_id in program.transfer_steps[transfer_id]
            if program.steps_by_id[step_id].xml_type is EndpointType.SEND
        )
        send_step = program.steps_by_id[send_step_id]
        threadblock = program.threadblock_for_step(send_step_id)
        preceding_nops = [
            step
            for step in threadblock.steps[: threadblock.steps.index(send_step)]
            if step.xml_type is EndpointType.NOP
            and step.node_id == transfer_id
        ]
        assert len(preceding_nops) == 2
        dependency = program.steps_by_id[send_step.dependency_step_id]
        assert dependency.transfer_id == "reduce-star-3"


def test_referenced_predecessor_steps_are_marked_for_completion_flags():
    schedule = allreduce_star_schedule()
    buffers = build_buffer_plan(
        schedule,
        resolved(CollectiveKind.ALL_REDUCE, ranks=4, slices=1),
    )
    endpoints = lower_endpoints(schedule, buffers)

    program = schedule_threadblocks(
        endpoints,
        build_transfer_dag(endpoints, schedule, buffers),
    )

    referenced = {
        step.dependency_step_id
        for step in program.steps_by_id.values()
        if step.dependency_step_id is not None
    }
    assert referenced == program.referenced_step_ids
    assert all(program.steps_by_id[step_id].has_dependence for step_id in referenced)
