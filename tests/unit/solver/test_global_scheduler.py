from dataclasses import replace

import pytest

from vericcl.errors import SemanticError
from vericcl.semantics.atom import Atom, PathStage, Schedule, Symbol, Transfer
from vericcl.solver.global_scheduler import assign_global_resources

from tests.unit.verification.simulator_helpers import (
    curve,
    opposite_direction_schedule,
    simulation_topology,
)


def _transfer(
    transfer_id,
    *,
    slice_id,
    slice_count,
    path,
    predecessors=(),
    st_time=0.0,
    ed_time=1.0,
):
    stages = tuple(
        PathStage(
            stage_id,
            kind,
            tuple(Symbol(src, dst, ready) for src, dst, ready in symbols),
        )
        for stage_id, kind, symbols in path
    )
    current = stages[-1]
    symbol = current.symbols[-1]
    atom = Atom(
        slice_id=slice_id,
        slice_size_bytes=1024,
        path=stages,
        st_time=st_time,
        ed_time=ed_time,
    )
    return Transfer(
        transfer_id=transfer_id,
        kind=current.operator,
        src_rank=symbol.src_rank,
        dst_rank=symbol.dst_rank,
        channel=7,
        stage_id=current.stage_id,
        member_slice_ids=frozenset({slice_id}),
        atoms=(atom,),
        st_time=st_time,
        ed_time=ed_time,
        predecessor_ids=frozenset(predecessors),
    )


def _schedule(
    transfers,
    *,
    rank_count,
    slice_count,
    semantic=None,
    metadata=None,
):
    values = tuple(transfers)
    semantic = (
        {
            transfer.transfer_id: tuple(sorted(transfer.predecessor_ids))
            for transfer in values
        }
        if semantic is None
        else semantic
    )
    schedule_metadata = {
        "path_scope": "global",
        "routing_only": True,
        "semantic_predecessors": semantic,
        "resource_slots": {
            transfer.transfer_id: {} for transfer in values
        },
    }
    if metadata:
        schedule_metadata.update(metadata)
    return Schedule(
        schedule_id="global-scheduler-fixture",
        transfers=values,
        final_state_ids=(),
        rank_count=rank_count,
        slice_count=slice_count,
        slice_size_bytes=1024,
        metadata=schedule_metadata,
    )


def _pipeline_schedule(*, reverse=False, shifted=False):
    first_start = 17.0 if shifted else 0.0
    first = _transfer(
        "stage-0-logical-0",
        slice_id=0,
        slice_count=2,
        path=((0, "SEND", ((0, 1, first_start),)),),
        st_time=first_start,
        ed_time=first_start + 11.0,
    )
    other_start = 31.0 if shifted else 9.0
    other = _transfer(
        "stage-0-logical-1",
        slice_id=1,
        slice_count=2,
        path=((0, "SEND", ((0, 1, other_start),)),),
        st_time=other_start,
        ed_time=other_start + 23.0,
    )
    downstream_start = 53.0 if shifted else 19.0
    downstream = _transfer(
        "stage-1-logical-0",
        slice_id=0,
        slice_count=2,
        path=(
            (0, "SEND", ((0, 1, first_start),)),
            (1, "SEND", ((1, 2, downstream_start),)),
        ),
        predecessors=(first.transfer_id,),
        st_time=downstream_start,
        ed_time=downstream_start + 29.0,
    )
    values = (first, other, downstream)
    if reverse:
        values = tuple(reversed(values))
    semantic = {
        first.transfer_id: (),
        other.transfer_id: (),
        downstream.transfer_id: (first.transfer_id,),
    }
    return _schedule(
        values,
        rank_count=3,
        slice_count=2,
        semantic=semantic,
    )


def test_opposite_directed_links_can_run_in_parallel():
    topology = simulation_topology(
        2,
        {
            (0, 1): curve(),
            (1, 0): curve(),
        },
        max_channels=1,
    )

    result = assign_global_resources(
        opposite_direction_schedule(),
        topology,
        1,
    )

    assert {transfer.st_time for transfer in result.transfers} == {0.0}
    assert {transfer.ed_time for transfer in result.transfers} == {2.0}


def test_same_directed_link_uses_parallel_channels_without_lane_overlap():
    transfers = tuple(
        _transfer(
            "send-{}".format(index),
            slice_id=index,
            slice_count=3,
            path=((0, "SEND", ((0, 1, float(10 - index)),)),),
            st_time=float(10 - index),
            ed_time=float(20 + index),
        )
        for index in range(3)
    )
    schedule = _schedule(
        tuple(reversed(transfers)),
        rank_count=2,
        slice_count=3,
    )
    topology = simulation_topology(
        2,
        {(0, 1): curve()},
        max_channels=4,
    )

    result = assign_global_resources(schedule, topology, 2)
    by_id = {transfer.transfer_id: transfer for transfer in result.transfers}

    assert (by_id["send-0"].channel, by_id["send-0"].st_time) == (0, 0.0)
    assert (by_id["send-1"].channel, by_id["send-1"].st_time) == (1, 0.0)
    assert (by_id["send-2"].channel, by_id["send-2"].st_time) == (0, 3.0)
    for channel in (0, 1):
        intervals = sorted(
            (transfer.st_time, transfer.ed_time)
            for transfer in result.transfers
            if transfer.channel == channel
        )
        assert all(
            previous[1] <= current[0]
            for previous, current in zip(intervals, intervals[1:])
        )


def test_shared_resource_slots_are_independent_from_logical_links():
    first = _transfer(
        "branch-a",
        slice_id=0,
        slice_count=2,
        path=((0, "SEND", ((0, 1, 8.0),)),),
        st_time=8.0,
        ed_time=19.0,
    )
    second = _transfer(
        "branch-b",
        slice_id=1,
        slice_count=2,
        path=((0, "SEND", ((0, 2, 4.0),)),),
        st_time=4.0,
        ed_time=23.0,
    )
    schedule = _schedule(
        (second, first),
        rank_count=3,
        slice_count=2,
    )
    topology = simulation_topology(
        3,
        {
            (0, 1): curve(),
            (0, 2): curve(),
        },
        max_channels=2,
        shared_links=((0, 1), (0, 2)),
        shared_channels=2,
    )

    result = assign_global_resources(schedule, topology, 2)
    slots = result.metadata["resource_slots"]

    assert {transfer.st_time for transfer in result.transfers} == {0.0}
    assert {slots[transfer.transfer_id]["nic"] for transfer in result.transfers} == {
        0,
        1,
    }


def test_fixed_channel_count_duration_does_not_speed_up_sparse_ready_set():
    topology = simulation_topology(
        2,
        {
            (0, 1): curve(),
            (1, 0): curve(),
        },
        max_channels=4,
    )

    result = assign_global_resources(
        opposite_direction_schedule(),
        topology,
        4,
    )

    assert {transfer.st_time for transfer in result.transfers} == {0.0}
    assert {transfer.ed_time for transfer in result.transfers} == {5.0}


def test_all_semantic_predecessors_must_finish_before_transfer_is_ready():
    first = _transfer(
        "first-parent",
        slice_id=0,
        slice_count=1,
        path=((0, "SEND", ((0, 1, 0.0),)),),
    )
    second = _transfer(
        "second-parent",
        slice_id=2,
        slice_count=1,
        path=((0, "SEND", ((2, 1, 0.0),)),),
    )
    child = _transfer(
        "child",
        slice_id=0,
        slice_count=1,
        path=(
            (0, "SEND", ((0, 1, 0.0),)),
            (1, "SEND", ((1, 3, 40.0),)),
        ),
        predecessors=(first.transfer_id, second.transfer_id),
        st_time=40.0,
        ed_time=90.0,
    )
    schedule = _schedule(
        (child, second, first),
        rank_count=4,
        slice_count=1,
    )
    topology = simulation_topology(
        4,
        {
            (0, 1): curve(),
            (2, 1): curve(),
            (1, 3): curve(),
        },
        max_channels=2,
    )

    result = assign_global_resources(schedule, topology, 2)
    by_id = {transfer.transfer_id: transfer for transfer in result.transfers}

    assert by_id["first-parent"].ed_time == 3.0
    assert by_id["second-parent"].ed_time == 3.0
    assert by_id["child"].st_time == 3.0
    assert {"first-parent", "second-parent"} <= by_id["child"].predecessor_ids


def test_ready_stage_one_transfer_overlaps_unrelated_stage_zero_slice():
    topology = simulation_topology(
        3,
        {
            (0, 1): curve(),
            (1, 2): curve(),
        },
        max_channels=1,
    )

    result = assign_global_resources(_pipeline_schedule(), topology, 1)
    by_id = {transfer.transfer_id: transfer for transfer in result.transfers}

    assert by_id["stage-1-logical-0"].st_time == 2.0
    assert by_id["stage-1-logical-0"].st_time < by_id[
        "stage-0-logical-1"
    ].ed_time


def test_provisional_times_and_transfer_order_do_not_change_result():
    topology = simulation_topology(
        3,
        {
            (0, 1): curve(),
            (1, 2): curve(),
        },
        max_channels=1,
    )

    first = assign_global_resources(_pipeline_schedule(), topology, 1)
    second = assign_global_resources(
        _pipeline_schedule(reverse=True, shifted=True),
        topology,
        1,
    )

    assert first == second
    assert "routing_only" not in first.metadata


def test_route_priority_precedes_stage_and_logical_position_ties():
    logical_zero = _transfer(
        "logical-zero",
        slice_id=0,
        slice_count=2,
        path=((0, "SEND", ((0, 1, 0.0),)),),
    )
    logical_one = _transfer(
        "logical-one",
        slice_id=1,
        slice_count=2,
        path=((0, "SEND", ((0, 1, 0.0),)),),
    )
    schedule = _schedule(
        (logical_zero, logical_one),
        rank_count=2,
        slice_count=2,
        metadata={
            "route_priorities": {
                logical_zero.transfer_id: 1,
                logical_one.transfer_id: 0,
            }
        },
    )
    topology = simulation_topology(
        2,
        {(0, 1): curve()},
        max_channels=1,
    )

    result = assign_global_resources(schedule, topology, 1)
    by_id = {transfer.transfer_id: transfer for transfer in result.transfers}

    assert by_id["logical-one"].st_time == 0.0
    assert by_id["logical-zero"].st_time == 2.0


def test_default_route_priority_defers_to_logical_position():
    logical_zero = _transfer(
        "logical-zero",
        slice_id=0,
        slice_count=2,
        path=((0, "SEND", ((0, 1, 0.0),)),),
    )
    logical_one = _transfer(
        "logical-one",
        slice_id=1,
        slice_count=2,
        path=((0, "SEND", ((0, 1, 0.0),)),),
    )
    schedule = _schedule(
        (logical_one, logical_zero),
        rank_count=2,
        slice_count=2,
    )
    topology = simulation_topology(
        2,
        {(0, 1): curve()},
        max_channels=1,
    )

    result = assign_global_resources(schedule, topology, 1)
    by_id = {transfer.transfer_id: transfer for transfer in result.transfers}

    assert by_id["logical-zero"].st_time == 0.0
    assert by_id["logical-one"].st_time == 2.0


@pytest.mark.parametrize(
    ("priorities", "message"),
    (
        ((), "must be a mapping"),
        ({"missing": 0}, "missing transfer"),
        ({"forward": True}, "non-negative integer"),
        ({"forward": 1.0}, "non-negative integer"),
        ({"forward": -1}, "non-negative integer"),
    ),
)
def test_scheduler_rejects_invalid_route_priorities(priorities, message):
    schedule = opposite_direction_schedule()
    metadata = dict(schedule.metadata)
    metadata["route_priorities"] = priorities
    schedule = replace(schedule, metadata=metadata)
    topology = simulation_topology(
        2,
        {
            (0, 1): curve(),
            (1, 0): curve(),
        },
    )

    with pytest.raises(SemanticError, match=message):
        assign_global_resources(schedule, topology, 1)


@pytest.mark.parametrize("channel_count", (0, -1, True, 1.0))
def test_scheduler_rejects_invalid_channel_count(channel_count):
    topology = simulation_topology(
        2,
        {
            (0, 1): curve(),
            (1, 0): curve(),
        },
    )

    with pytest.raises(SemanticError, match="channel_count"):
        assign_global_resources(
            opposite_direction_schedule(),
            topology,
            channel_count,
        )


@pytest.mark.parametrize(
    ("semantic", "message"),
    (
        ("invalid", "mapping"),
        ({"forward": ()}, "cover every transfer"),
        (
            {"forward": ("missing",), "reverse": ()},
            "missing",
        ),
        (
            {"forward": ("reverse",), "reverse": ("forward",)},
            "cycle",
        ),
    ),
)
def test_scheduler_rejects_invalid_semantic_dag(semantic, message):
    schedule = opposite_direction_schedule()
    metadata = dict(schedule.metadata)
    metadata["semantic_predecessors"] = semantic
    schedule = replace(schedule, metadata=metadata)
    topology = simulation_topology(
        2,
        {
            (0, 1): curve(),
            (1, 0): curve(),
        },
    )

    with pytest.raises(SemanticError, match=message):
        assign_global_resources(schedule, topology, 1)


def test_scheduler_rejects_missing_topology_link_without_partial_result():
    topology = simulation_topology(
        2,
        {(0, 1): curve()},
    )

    with pytest.raises(SemanticError, match="topology"):
        assign_global_resources(
            opposite_direction_schedule(),
            topology,
            1,
        )


def test_scheduler_rejects_incomplete_atom_path_schedule():
    schedule = _pipeline_schedule()
    child = next(
        transfer
        for transfer in schedule.transfers
        if transfer.transfer_id == "stage-1-logical-0"
    )
    child = replace(child, predecessor_ids=frozenset())
    incomplete = _schedule(
        (child,),
        rank_count=3,
        slice_count=2,
        semantic={child.transfer_id: ()},
    )
    topology = simulation_topology(
        3,
        {(1, 2): curve()},
        max_channels=1,
    )

    with pytest.raises(SemanticError, match="operation is missing"):
        assign_global_resources(incomplete, topology, 1)
