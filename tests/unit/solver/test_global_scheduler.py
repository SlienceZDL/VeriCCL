import pytest

from vericcl.composer.dual import reverse_allgather_schedule
from vericcl.errors import SemanticError
from vericcl.semantics.atom import Atom, PathStage, Schedule, Symbol, Transfer
from vericcl.solver.global_scheduler import assign_global_resources
from vericcl.topology.model import (
    DirectedLink,
    LinkKey,
    PerformanceCurve,
    SharedResource,
    Topology,
)
from vericcl.verification.simulator import simulate_schedule

from tests.unit.composer.helpers import (
    reduce_spec,
    reduce_target,
    virtual_reduce_star,
)


pytestmark = pytest.mark.phase03


def _topology(*links, max_channels=2, resource_channels=None):
    curve = PerformanceCurve(1.0, 2.0, {})
    keys = tuple(LinkKey(*link) for link in links)
    resource_ids = ("nic",) if resource_channels is not None else ()
    return Topology(
        rank_count=max(rank for link in links for rank in link) + 1,
        links={
            key: DirectedLink(key, max_channels, curve, resource_ids)
            for key in keys
        },
        shared_resources=(
            {
                "nic": SharedResource(
                    "nic",
                    keys,
                    resource_channels,
                    curve,
                )
            }
            if resource_channels is not None
            else {}
        ),
        node_membership={
            rank: 0 for rank in range(max(rank for link in links for rank in link) + 1)
        },
        gateways=frozenset(),
        warnings=(),
    )


def _transfer(
    transfer_id,
    src_rank,
    dst_rank,
    slice_id,
    *,
    path=None,
    predecessors=(),
    channel=1,
):
    symbols = (
        tuple(path)
        if path is not None
        else (Symbol(src_rank, dst_rank, 8.0),)
    )
    atom = Atom(
        slice_id=slice_id,
        slice_size_bytes=1024,
        path=(PathStage(0, "SEND", symbols),),
        st_time=9.0,
        ed_time=10.0,
    )
    return Transfer(
        transfer_id=transfer_id,
        kind="SEND",
        src_rank=src_rank,
        dst_rank=dst_rank,
        channel=channel,
        stage_id=0,
        member_slice_ids=frozenset({slice_id}),
        atoms=(atom,),
        st_time=9.0,
        ed_time=10.0,
        predecessor_ids=frozenset(predecessors),
    )


def _schedule(transfers, semantic=None, resource_slots=None):
    transfers = tuple(transfers)
    return Schedule(
        schedule_id="routing-only",
        transfers=transfers,
        final_state_ids=(),
        rank_count=max(
            rank
            for transfer in transfers
            for rank in (transfer.src_rank, transfer.dst_rank)
        )
        + 1,
        slice_count=2,
        slice_size_bytes=1024,
        metadata={
            "path_scope": "global",
            "routing_only": True,
            "semantic_predecessors": (
                {
                    transfer.transfer_id: tuple(transfer.predecessor_ids)
                    for transfer in transfers
                }
                if semantic is None
                else semantic
            ),
            "resource_slots": (
                {
                    transfer.transfer_id: {"nic": 1}
                    for transfer in transfers
                }
                if resource_slots is None
                else resource_slots
            ),
        },
    )


def _by_id(schedule):
    return {
        transfer.transfer_id: transfer for transfer in schedule.transfers
    }


def test_reverse_directions_overlap_but_one_lane_serializes():
    reverse = _schedule(
        (
            _transfer("forward", 0, 1, 0),
            _transfer("reverse", 1, 0, 2),
        ),
        resource_slots={"forward": {}, "reverse": {}},
    )

    overlapping = assign_global_resources(
        reverse,
        _topology((0, 1), (1, 0), max_channels=1),
        1,
    )

    assert {transfer.st_time for transfer in overlapping.transfers} == {0.0}

    same_lane = _schedule(
        (
            _transfer("same-a", 0, 1, 0),
            _transfer("same-b", 0, 1, 1),
        ),
        resource_slots={"same-a": {}, "same-b": {}},
    )
    serialized = assign_global_resources(
        same_lane,
        _topology((0, 1), max_channels=1),
        1,
    )
    by_id = _by_id(serialized)

    assert by_id["same-a"].ed_time <= by_id["same-b"].st_time
    assert "same-a" in by_id["same-b"].predecessor_ids


def test_distinct_channels_and_distinct_resource_slots_overlap():
    schedule = _schedule(
        (
            _transfer("parallel-a", 0, 1, 0),
            _transfer("parallel-b", 0, 1, 1),
        )
    )

    assigned = assign_global_resources(
        schedule,
        _topology(
            (0, 1),
            max_channels=2,
            resource_channels=2,
        ),
        2,
    )
    by_id = _by_id(assigned)

    assert {transfer.st_time for transfer in assigned.transfers} == {0.0}
    assert {transfer.channel for transfer in assigned.transfers} == {0, 1}
    assert {
        assigned.metadata["resource_slots"][transfer.transfer_id]["nic"]
        for transfer in assigned.transfers
    } == {0, 1}
    assert not by_id["parallel-a"].predecessor_ids
    assert not by_id["parallel-b"].predecessor_ids


def test_shared_resource_serializes_only_one_shared_slot():
    schedule = _schedule(
        (
            _transfer("resource-a", 0, 1, 0),
            _transfer("resource-b", 0, 2, 1),
        )
    )

    assigned = assign_global_resources(
        schedule,
        _topology(
            (0, 1),
            (0, 2),
            max_channels=1,
            resource_channels=1,
        ),
        1,
    )
    by_id = _by_id(assigned)

    assert by_id["resource-a"].ed_time <= by_id["resource-b"].st_time
    assert "resource-a" in by_id["resource-b"].predecessor_ids


def test_shared_resource_duration_covers_concurrency_across_distinct_links():
    link_curve = PerformanceCurve(1.0, 2.0, {})
    resource_curve = PerformanceCurve(1.0, 4.0, {})
    keys = (LinkKey(0, 1), LinkKey(0, 2))
    topology = Topology(
        rank_count=3,
        links={
            key: DirectedLink(key, 1, link_curve, ("nic",))
            for key in keys
        },
        shared_resources={
            "nic": SharedResource("nic", keys, 2, resource_curve),
        },
        node_membership={0: 0, 1: 0, 2: 0},
        gateways=frozenset(),
        warnings=(),
    )
    schedule = _schedule(
        (
            _transfer("shared-a", 0, 1, 0),
            _transfer("shared-b", 0, 2, 1),
        )
    )

    assigned = assign_global_resources(schedule, topology, 2)
    simulated = simulate_schedule(assigned, topology)

    assert {transfer.st_time for transfer in assigned.transfers} == {0.0}
    assert {transfer.ed_time for transfer in assigned.transfers} == {7.0}
    assert simulated.end_times == {
        "shared-a": 7.0,
        "shared-b": 7.0,
    }


def test_independent_resources_do_not_create_false_serialization():
    curve = PerformanceCurve(1.0, 2.0, {})
    first = LinkKey(0, 1)
    second = LinkKey(0, 2)
    topology = Topology(
        rank_count=3,
        links={
            first: DirectedLink(first, 1, curve, ("nic-a",)),
            second: DirectedLink(second, 1, curve, ("nic-b",)),
        },
        shared_resources={
            "nic-a": SharedResource("nic-a", (first,), 1, curve),
            "nic-b": SharedResource("nic-b", (second,), 1, curve),
        },
        node_membership={0: 0, 1: 0, 2: 0},
        gateways=frozenset(),
        warnings=(),
    )
    schedule = _schedule(
        (
            _transfer("independent-a", 0, 1, 0),
            _transfer("independent-b", 0, 2, 1),
        )
    )

    assigned = assign_global_resources(schedule, topology, 1)

    assert {transfer.st_time for transfer in assigned.transfers} == {0.0}
    assert assigned.metadata["resource_slots"] == {
        "independent-a": {"nic-a": 0},
        "independent-b": {"nic-b": 0},
    }


def test_scheduler_rebuilds_semantic_ready_times_and_provisional_fields():
    first = _transfer(
        "chain-a",
        0,
        1,
        0,
        predecessors=("chain-b",),
    )
    second = _transfer(
        "chain-b",
        1,
        2,
        0,
        path=(Symbol(0, 1, 8.0), Symbol(1, 2, 8.0)),
        predecessors=("chain-a",),
    )
    schedule = _schedule(
        (second, first),
        semantic={"chain-a": (), "chain-b": ("chain-a",)},
        resource_slots={"chain-a": {}, "chain-b": {}},
    )

    assigned = assign_global_resources(
        schedule,
        _topology((0, 1), (1, 2), max_channels=1),
        1,
    )
    by_id = _by_id(assigned)

    assert by_id["chain-a"].channel == 0
    assert by_id["chain-a"].st_time == 0.0
    assert not by_id["chain-a"].predecessor_ids
    assert by_id["chain-b"].st_time == by_id["chain-a"].ed_time
    assert by_id["chain-a"].transfer_id in by_id["chain-b"].predecessor_ids
    assert [
        symbol.ready_time
        for symbol in by_id["chain-b"].atoms[0].path[0].symbols
    ] == [0.0, by_id["chain-a"].ed_time]
    assert assigned.metadata["resource_slots"] == {
        "chain-a": {},
        "chain-b": {},
    }


def test_scheduler_preserves_versioned_aggregate_state_fan_in():
    reduced = reverse_allgather_schedule(
        virtual_reduce_star(3),
        reduce_spec(),
        reduce_target(3),
    )
    assigned = assign_global_resources(
        reduced,
        _topology((1, 0), (2, 0), max_channels=1),
        1,
    )
    ordered = sorted(
        assigned.transfers,
        key=lambda transfer: (transfer.st_time, transfer.transfer_id),
    )

    assert ordered[0].ed_time <= ordered[1].st_time
    assert ordered[0].transfer_id in ordered[1].predecessor_ids
    assert frozenset(
        assigned.metadata["tree_contributors"][ordered[1].transfer_id]
    ) == frozenset({0, 1, 2})


def test_scheduler_is_independent_of_provisional_transfer_order():
    transfers = (
        _transfer("ordered-a", 0, 1, 0),
        _transfer("ordered-b", 0, 1, 1),
    )
    topology = _topology((0, 1), max_channels=2)
    forward = assign_global_resources(
        _schedule(
            transfers,
            resource_slots={"ordered-a": {}, "ordered-b": {}},
        ),
        topology,
        2,
    )
    backward = assign_global_resources(
        _schedule(
            reversed(transfers),
            resource_slots={"ordered-a": {}, "ordered-b": {}},
        ),
        topology,
        2,
    )

    assert backward == forward


def test_scheduler_reports_cycles_and_unavailable_capacity():
    transfers = (
        _transfer("cycle-a", 0, 1, 0, predecessors=("cycle-b",)),
        _transfer("cycle-b", 0, 1, 1, predecessors=("cycle-a",)),
    )
    schedule = _schedule(
        transfers,
        resource_slots={"cycle-a": {}, "cycle-b": {}},
    )
    topology = _topology((0, 1), max_channels=1)

    with pytest.raises(SemanticError, match="cycle"):
        assign_global_resources(schedule, topology, 1)
    with pytest.raises(SemanticError, match="channel_count"):
        assign_global_resources(schedule, topology, 0)


def test_scheduler_rejects_a_transfer_missing_from_the_topology():
    schedule = _schedule(
        (_transfer("missing", 0, 2, 0),),
        resource_slots={"missing": {}},
    )

    with pytest.raises(SemanticError, match="topology"):
        assign_global_resources(
            schedule,
            _topology((0, 1), max_channels=1),
            1,
        )
