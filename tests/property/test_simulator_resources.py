import pytest
from hypothesis import given, settings, strategies as st

from vericcl.solver.global_scheduler import assign_global_resources
from vericcl.topology.model import LinkKey
from vericcl.topology.performance import transfer_duration_us
from vericcl.verification.resource_events import directed_link_timeline_id
from vericcl.verification.simulator import simulate_schedule

from tests.unit.verification.simulator_helpers import (
    curve,
    same_direction_schedule,
    simulation_topology,
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

    scheduled = assign_global_resources(
        same_direction_schedule(channel_count),
        topology,
        channel_count,
    )
    result = simulate_schedule(scheduled, topology)

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
