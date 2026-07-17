from dataclasses import replace

import pytest

from vericcl.errors import SemanticError
from vericcl.topology.model import LinkKey
from vericcl.topology.performance import transfer_duration_us
from vericcl.verification.resource_events import (
    ResourceInterval,
    ResourceTimeline,
    directed_link_timeline_id,
    shared_resource_timeline_id,
    timeline_resource_ids,
)
from vericcl.verification.simulator import (
    SimulationEvent,
    simulate_schedule,
)

from tests.unit.verification.simulator_helpers import (
    curve,
    opposite_direction_schedule,
    relay_schedule,
    same_direction_schedule,
    schedule,
    simulation_topology,
    transfer,
)


pytestmark = pytest.mark.phase05


def test_opposite_directions_progress_in_parallel():
    topology = simulation_topology(
        2,
        {(0, 1): curve(), (1, 0): curve()},
    )

    result = simulate_schedule(opposite_direction_schedule(), topology)

    expected = transfer_duration_us(topology.link(LinkKey(0, 1)), 1024, 1)
    assert result.completion_time_us == pytest.approx(expected)
    assert result.start_times["forward"] == pytest.approx(0.0)
    assert result.start_times["reverse"] == pytest.approx(0.0)


def test_two_channels_share_total_directed_link_bandwidth():
    topology = simulation_topology(2, {(0, 1): curve()})

    result = simulate_schedule(same_direction_schedule(2), topology)

    expected = transfer_duration_us(topology.link(LinkKey(0, 1)), 1024, 2)
    assert result.completion_time_us == pytest.approx(expected)
    timeline = result.timelines[directed_link_timeline_id(LinkKey(0, 1))]
    assert max(interval.concurrency for interval in timeline.intervals) == 2


def test_equal_time_events_use_stable_transfer_id_order():
    topology = simulation_topology(
        2,
        {(0, 1): curve(), (1, 0): curve()},
    )

    result = simulate_schedule(opposite_direction_schedule(), topology)
    completed = [
        event.transfer_id
        for event in result.events
        if event.event_type == "complete"
    ]

    assert completed == ["forward", "reverse"]


def test_reduce_join_uses_maximum_predecessor_completion():
    contributions = (
        transfer("reduce-fast", 1, 0, 0, 1, 1, kind="REDUCE"),
        transfer("reduce-slow", 2, 0, 0, 2, 1, kind="REDUCE"),
    )
    joined = transfer(
        "joined-send",
        0,
        3,
        0,
        0,
        1,
        predecessors=("reduce-fast", "reduce-slow"),
        st_time=1.0,
        ed_time=2.0,
    )
    value = schedule(
        "reduce-join",
        4,
        1,
        contributions + (joined,),
    )
    topology = simulation_topology(
        4,
        {
            (1, 0): curve(1.0, 2.0),
            (2, 0): curve(1.0, 4.0),
            (0, 3): curve(1.0, 2.0),
        },
    )

    result = simulate_schedule(value, topology)

    assert result.start_times["joined-send"] == pytest.approx(
        result.end_times["reduce-slow"]
    )
    assert result.semantic_ready_times["joined-send"] == pytest.approx(
        max(
            result.end_times["reduce-fast"],
            result.end_times["reduce-slow"],
        )
    )


def test_shared_nic_uses_cross_link_concurrency():
    transfers = (
        transfer("nic-a", 0, 1, 0, 0, 1),
        transfer("nic-b", 2, 3, 0, 2, 1),
    )
    value = schedule(
        "shared-nic",
        4,
        1,
        transfers,
        resource_slots={"nic-a": {"nic": 0}, "nic-b": {"nic": 1}},
    )
    topology = simulation_topology(
        4,
        {(0, 1): curve(), (2, 3): curve()},
        shared_links=((0, 1), (2, 3)),
        shared_curve=curve(1.0, 3.0),
        shared_channels=2,
    )

    result = simulate_schedule(value, topology)

    timeline = result.timelines[shared_resource_timeline_id("nic")]
    assert max(interval.concurrency for interval in timeline.intervals) == 2
    assert result.completion_time_us == pytest.approx(5.0)


def test_idle_lane_waits_for_semantic_data():
    topology = simulation_topology(
        3,
        {(0, 1): curve(), (1, 2): curve()},
    )

    result = simulate_schedule(relay_schedule(), topology)

    assert result.start_times["relay-second"] == pytest.approx(
        result.end_times["relay-first"]
    )
    timeline = result.timelines[directed_link_timeline_id(LinkKey(1, 2))]
    assert timeline.idle_intervals[0].start_time_us == pytest.approx(0.0)
    assert timeline.idle_intervals[0].end_time_us == pytest.approx(
        result.end_times["relay-first"]
    )


def test_active_concurrency_change_recomputes_remaining_duration():
    long = transfer("long", 0, 1, 0, 0, 2)
    trigger = transfer("trigger", 2, 3, 0, 4, 2)
    late = transfer(
        "late",
        0,
        1,
        1,
        1,
        2,
        predecessors=("trigger",),
        st_time=1.0,
        ed_time=2.0,
    )
    value = schedule("dynamic-k", 4, 2, (long, trigger, late))
    topology = simulation_topology(
        4,
        {
            (0, 1): curve(0.0, 10.0),
            (2, 3): curve(0.0, 2.0),
        },
    )

    result = simulate_schedule(value, topology)

    timeline = result.timelines[directed_link_timeline_id(LinkKey(0, 1))]
    assert {interval.concurrency for interval in timeline.busy_intervals} == {
        1,
        2,
    }
    assert result.end_times["long"] > 10.0
    assert result.end_times["late"] > result.end_times["long"]


def test_member_atoms_are_counted_as_one_physical_transfer():
    value = same_direction_schedule(1)
    original = value.transfers[0]
    duplicate_atom = original.atoms[0]
    object.__setattr__(original, "atoms", (duplicate_atom, duplicate_atom))

    topology = simulation_topology(2, {(0, 1): curve()})
    result = simulate_schedule(value, topology)

    assert tuple(result.start_times) == ("send-0",)
    assert len([event for event in result.events if event.event_type == "start"]) == 1


def test_calibrated_shared_resource_curve_is_used():
    value = schedule(
        "calibrated",
        2,
        1,
        (transfer("calibrated-send", 0, 1, 0, 0, 1),),
        resource_slots={"calibrated-send": {"nic": 0}},
    )
    calibrated = curve(1.0, 2.0, {1: 1024.0})
    topology = simulation_topology(
        2,
        {(0, 1): calibrated},
        shared_links=((0, 1),),
        shared_curve=calibrated,
        shared_channels=1,
    )

    result = simulate_schedule(value, topology)

    assert result.completion_time_us == pytest.approx(2.0)


def test_simulator_rejects_invalid_inputs_and_topology_geometry():
    value = same_direction_schedule(1)
    valid_topology = simulation_topology(2, {(0, 1): curve()})
    with pytest.raises(SemanticError, match="Schedule"):
        simulate_schedule(None, valid_topology)
    with pytest.raises(SemanticError, match="Topology"):
        simulate_schedule(value, None)
    with pytest.raises(SemanticError, match="rank counts"):
        simulate_schedule(
            value,
            simulation_topology(3, {(0, 1): curve()}),
        )
    with pytest.raises(SemanticError, match="missing link"):
        simulate_schedule(
            value,
            simulation_topology(2, {(1, 0): curve()}),
        )
    with pytest.raises(SemanticError, match="channel"):
        simulate_schedule(
            same_direction_schedule(2),
            simulation_topology(
                2,
                {(0, 1): curve()},
                max_channels=1,
            ),
        )


def test_simulator_rejects_invalid_resource_and_dependency_metadata():
    value = same_direction_schedule(1)
    metadata = dict(value.metadata)
    metadata["semantic_predecessors"] = "invalid"
    with pytest.raises(SemanticError, match="semantic_predecessors"):
        simulate_schedule(
            replace(value, metadata=metadata),
            simulation_topology(2, {(0, 1): curve()}),
        )

    shared = simulation_topology(
        2,
        {(0, 1): curve()},
        shared_links=((0, 1),),
        shared_channels=1,
    )
    with pytest.raises(SemanticError, match="do not match"):
        simulate_schedule(value, shared)
    metadata = dict(value.metadata)
    metadata["resource_slots"] = {"send-0": {"nic": 1}}
    with pytest.raises(SemanticError, match="slot"):
        simulate_schedule(replace(value, metadata=metadata), shared)


def test_dependency_cycle_cannot_make_progress():
    value = opposite_direction_schedule()
    forward = replace(
        value.transfers[0],
        predecessor_ids=frozenset({"reverse"}),
    )
    reverse = replace(
        value.transfers[1],
        predecessor_ids=frozenset({"forward"}),
    )
    metadata = dict(value.metadata)
    metadata["semantic_predecessors"] = {
        "forward": ("reverse",),
        "reverse": ("forward",),
    }
    cyclic = replace(
        value,
        transfers=(forward, reverse),
        metadata=metadata,
    )
    topology = simulation_topology(
        2,
        {(0, 1): curve(), (1, 0): curve()},
    )

    with pytest.raises(SemanticError, match="cannot make progress"):
        simulate_schedule(cyclic, topology)


def test_resource_event_models_reject_invalid_records():
    with pytest.raises(SemanticError, match="LinkKey"):
        directed_link_timeline_id(None)
    with pytest.raises(SemanticError, match="resource_id"):
        shared_resource_timeline_id("")
    with pytest.raises(SemanticError, match="positive duration"):
        ResourceInterval("link:0->1", 1.0, 1.0, ())
    interval = ResourceInterval("link:0->1", 0.0, 1.0, ("tx",))
    with pytest.raises(SemanticError, match="intervals"):
        ResourceTimeline("other", (interval,))
    gap = ResourceInterval("link:0->1", 2.0, 3.0, ())
    with pytest.raises(SemanticError, match="contiguous"):
        ResourceTimeline("link:0->1", (interval, gap))
    with pytest.raises(SemanticError, match="Topology"):
        timeline_resource_ids(None)


def test_simulation_event_model_rejects_invalid_event_type():
    with pytest.raises(SemanticError, match="event type"):
        SimulationEvent(
            0.0,
            "invalid",
            "tx",
            next(iter(simulation_topology(2, {(0, 1): curve()}).lanes(
                LinkKey(0, 1),
                1,
            ))),
            1,
            (),
        )


def test_semantic_predecessor_is_a_start_gate():
    value = opposite_direction_schedule()
    metadata = dict(value.metadata)
    metadata["semantic_predecessors"] = {
        "forward": (),
        "reverse": ("forward",),
    }
    value = replace(value, metadata=metadata)
    topology = simulation_topology(
        2,
        {(0, 1): curve(), (1, 0): curve()},
    )

    result = simulate_schedule(value, topology)

    assert result.start_times["reverse"] == pytest.approx(
        result.end_times["forward"]
    )


def test_zero_duration_transfer_completes_without_zero_length_interval():
    topology = simulation_topology(
        2,
        {(0, 1): curve(0.0, 0.0)},
    )

    result = simulate_schedule(same_direction_schedule(1), topology)

    assert result.completion_time_us == pytest.approx(0.0)
    assert [event.event_type for event in result.events] == [
        "start",
        "complete",
    ]
    timeline = result.timelines[directed_link_timeline_id(LinkKey(0, 1))]
    assert timeline.intervals == ()
