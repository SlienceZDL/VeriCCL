from dataclasses import replace

import pytest
from hypothesis import given, settings, strategies as st

from vericcl.topology.model import LinkKey
from vericcl.topology.performance import transfer_duration_us
from vericcl.solver.global_scheduler import assign_global_resources
from vericcl.verification.resource_events import directed_link_timeline_id
from vericcl.verification.simulator import simulate_schedule

from tests.unit.verification.simulator_helpers import (
    curve,
    same_direction_schedule,
    schedule,
    simulation_topology,
    transfer,
)


pytestmark = pytest.mark.phase05


@settings(max_examples=12, deadline=None)
@given(channel_count=st.integers(min_value=1, max_value=4))
def test_same_link_concurrency_uses_one_physical_transfer_per_channel(
    channel_count,
):
    topology = simulation_topology(
        2,
        {(0, 1): curve()},
        max_channels=4,
    )

    result = simulate_schedule(
        same_direction_schedule(channel_count),
        topology,
    )

    expected = transfer_duration_us(
        topology.link(LinkKey(0, 1)),
        1024,
        channel_count,
    )
    timeline = result.timelines[directed_link_timeline_id(LinkKey(0, 1))]
    assert result.completion_time_us == pytest.approx(expected)
    assert max(interval.concurrency for interval in timeline.intervals) == (
        channel_count
    )
    assert len(result.start_times) == channel_count


@settings(max_examples=12, deadline=None)
@given(channel_count=st.integers(min_value=1, max_value=4))
def test_global_scheduler_allocations_replay_in_the_event_simulator(
    channel_count,
):
    topology = simulation_topology(
        2,
        {(0, 1): curve()},
        max_channels=4,
    )
    provisional = same_direction_schedule(channel_count)
    metadata = dict(provisional.metadata)
    metadata["routing_only"] = True
    assigned = assign_global_resources(
        replace(provisional, metadata=metadata),
        topology,
        channel_count,
    )

    result = simulate_schedule(assigned, topology)

    assert result.start_times == {
        transfer.transfer_id: transfer.st_time
        for transfer in assigned.transfers
    }
    assert result.end_times == pytest.approx(
        {
            transfer.transfer_id: transfer.ed_time
            for transfer in assigned.transfers
        }
    )


@settings(max_examples=12, deadline=None)
@given(channel_count=st.integers(min_value=1, max_value=4))
def test_shared_resource_assignments_replay_at_full_slot_concurrency(
    channel_count,
):
    topology = simulation_topology(
        3,
        {(0, 1): curve(), (0, 2): curve()},
        max_channels=4,
        shared_links=((0, 1), (0, 2)),
        shared_curve=curve(alpha=1.0, invbw=3.0),
        shared_channels=4,
    )
    transfers = tuple(
        transfer(
            "shared-{}".format(index),
            0,
            1 + index % 2,
            3,
            index,
            channel_count,
        )
        for index in range(channel_count)
    )
    provisional = schedule(
        "shared-routing-only",
        3,
        channel_count,
        transfers,
    )
    metadata = dict(provisional.metadata)
    metadata["routing_only"] = True
    assigned = assign_global_resources(
        replace(provisional, metadata=metadata),
        topology,
        channel_count,
    )

    result = simulate_schedule(assigned, topology)

    assert result.start_times == {
        transfer.transfer_id: transfer.st_time
        for transfer in assigned.transfers
    }
    assert result.end_times == pytest.approx(
        {
            transfer.transfer_id: transfer.ed_time
            for transfer in assigned.transfers
        }
    )
