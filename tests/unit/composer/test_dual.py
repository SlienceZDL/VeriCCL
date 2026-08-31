from dataclasses import replace

import pytest

from vericcl.composer.dual import reverse_allgather_schedule
from vericcl.errors import SemanticError
from vericcl.planner.dual import extract_dual_trees
from vericcl.planner.model import StageInterface
from vericcl.semantics.atom import Atom, PathStage, Schedule, Symbol, Transfer
from vericcl.semantics.collective import (
    CollectiveKind,
    CollectiveSpec,
    OutputSlot,
)
from vericcl.solver.constructive import construct_candidate

from tests.gurobi.helpers import reduction_dual_problem
from tests.unit.composer.helpers import (
    reduce_spec,
    reduce_target,
    virtual_reduce_chain,
    virtual_reduce_star,
    virtual_two_value_reduce,
)


pytestmark = pytest.mark.phase03


def test_ag_edge_becomes_reduce_with_rebuilt_state():
    problem = reduction_dual_problem()
    virtual = construct_candidate(problem, channel_count=1)

    reduced = reverse_allgather_schedule(
        virtual,
        problem.node.local_collective,
        problem.node.logical_output,
    )

    assert len(reduced.transfers) == 1
    transfer = reduced.transfers[0]
    assert transfer.kind == "REDUCE"
    assert (transfer.src_rank, transfer.dst_rank) == (1, 0)
    assert transfer.member_slice_ids == frozenset({8})
    assert {atom.slice_id for atom in transfer.atoms} == {8}
    assert transfer.physical_bytes == reduced.slice_size_bytes
    assert transfer.atoms[0].path[0].operator == "REDUCE"
    assert reduced.metadata["final_outputs"]["r00000000-o00000000"] == (
        0,
        8,
    )


def test_multihop_reversal_uses_postorder_dependencies_and_member_paths():
    reduced = reverse_allgather_schedule(
        virtual_reduce_chain(3),
        reduce_spec(),
        reduce_target(3),
    )
    by_edge = {
        (transfer.src_rank, transfer.dst_rank): transfer
        for transfer in reduced.transfers
    }
    leaf = by_edge[(2, 1)]
    root = by_edge[(1, 0)]

    assert leaf.ed_time <= root.st_time
    assert leaf.transfer_id in root.predecessor_ids
    assert root.member_slice_ids == frozenset({1, 2})
    member_path = next(atom for atom in root.atoms if atom.slice_id == 2)
    assert [
        (symbol.src_rank, symbol.dst_rank)
        for symbol in member_path.path[0].symbols
    ] == [(2, 1), (1, 0)]
    assert [
        symbol.ready_time for symbol in member_path.path[0].symbols
    ] == [0.0, leaf.ed_time]
    assert reduced.metadata["path_roots"][root.transfer_id] == {
        1: 1,
        2: 2,
    }


def test_dual_rejects_wrong_operator_or_target_contributors():
    virtual = virtual_reduce_chain(3)
    wrong_spec = CollectiveSpec(
        kind=CollectiveKind.BROADCAST,
        datatype="float32",
        root=0,
    )

    with pytest.raises(SemanticError, match="reduction"):
        reverse_allgather_schedule(virtual, wrong_spec, reduce_target(3))
    with pytest.raises(SemanticError, match="contributors"):
        reverse_allgather_schedule(
            virtual,
            reduce_spec(),
            reduce_target(2),
        )


def test_star_dual_keeps_independent_contributions_parallel():
    reduced = reverse_allgather_schedule(
        virtual_reduce_star(3),
        reduce_spec(),
        reduce_target(3),
    )

    assert {
        (transfer.src_rank, transfer.dst_rank)
        for transfer in reduced.transfers
    } == {(1, 0), (2, 0)}
    assert all(not transfer.predecessor_ids for transfer in reduced.transfers)
    assert {transfer.st_time for transfer in reduced.transfers} == {0.0}
    assert {
        tuple(sorted(transfer.member_slice_ids))
        for transfer in reduced.transfers
    } == {(1,), (2,)}


def test_non_routing_dual_matches_the_legacy_schedule_object_and_metadata():
    reduced = reverse_allgather_schedule(
        virtual_reduce_star(3),
        reduce_spec(),
        reduce_target(3),
    )
    first_id = "reduce-virtual-star-t0001"
    second_id = "reduce-virtual-star-t0002"
    expected = Schedule(
        schedule_id="virtual-star-reduce",
        transfers=(
            Transfer(
                transfer_id=first_id,
                kind="REDUCE",
                src_rank=1,
                dst_rank=0,
                channel=0,
                stage_id=0,
                member_slice_ids=frozenset({1}),
                atoms=(
                    Atom(
                        slice_id=1,
                        slice_size_bytes=1024,
                        path=(
                            PathStage(0, "REDUCE", (Symbol(1, 0, 0.0),)),
                        ),
                        st_time=0.0,
                        ed_time=2.0,
                    ),
                ),
                st_time=0.0,
                ed_time=2.0,
                predecessor_ids=frozenset(),
            ),
            Transfer(
                transfer_id=second_id,
                kind="REDUCE",
                src_rank=2,
                dst_rank=0,
                channel=0,
                stage_id=0,
                member_slice_ids=frozenset({2}),
                atoms=(
                    Atom(
                        slice_id=2,
                        slice_size_bytes=1024,
                        path=(
                            PathStage(0, "REDUCE", (Symbol(2, 0, 0.0),)),
                        ),
                        st_time=0.0,
                        ed_time=2.0,
                    ),
                ),
                st_time=0.0,
                ed_time=2.0,
                predecessor_ids=frozenset(),
            ),
        ),
        final_state_ids=("reduce-r00000000-o00000000",),
        rank_count=3,
        slice_count=1,
        slice_size_bytes=1024,
        metadata={
            "path_scope": "stage_suffix",
            "path_roots": {first_id: {1: 1}, second_id: {2: 2}},
            "reduction_dual": False,
            "dual_converted": True,
            "dual_source_schedule_id": "virtual-star",
            "reduction_op": "sum",
            "semantic_predecessors": {first_id: (), second_id: ()},
            "semantic_contributors": {first_id: (1,), second_id: (2,)},
            "tree_contributors": {
                first_id: (0, 1, 2),
                second_id: (0, 1, 2),
            },
            "resource_slots": {first_id: {}, second_id: {}},
            "final_outputs": {"r00000000-o00000000": (0, 1, 2)},
            "final_ready_times": {"r00000000-o00000000": 2.0},
        },
    )

    assert reduced == expected
    assert reduced.metadata == expected.metadata


def test_zero_edge_dual_preserves_already_local_state():
    virtual = Schedule(
        schedule_id="zero-edge-dual",
        transfers=(),
        final_state_ids=(),
        rank_count=1,
        slice_count=1,
        slice_size_bytes=1024,
        metadata={
            "path_scope": "stage_suffix",
            "path_roots": {},
            "reduction_dual": True,
            "semantic_contributors": {},
            "tree_contributors": {},
            "resource_slots": {},
        },
    )
    target = StageInterface({OutputSlot(0, 0): frozenset({0})})

    reduced = reverse_allgather_schedule(virtual, reduce_spec(), target)

    assert reduced.transfers == ()
    assert reduced.metadata["final_outputs"] == {
        "r00000000-o00000000": (0,)
    }
    assert reduced.metadata["final_ready_times"] == {
        "r00000000-o00000000": 0.0
    }


def test_dual_serializes_multiple_values_on_one_lane_and_resource_slot():
    target = StageInterface(
        {
            OutputSlot(0, 0): frozenset({0, 2}),
            OutputSlot(0, 1): frozenset({1, 3}),
        }
    )

    reduced = reverse_allgather_schedule(
        virtual_two_value_reduce(),
        reduce_spec(),
        target,
    )
    ordered = sorted(reduced.transfers, key=lambda item: item.st_time)

    assert ordered[0].ed_time <= ordered[1].st_time
    assert ordered[0].transfer_id in ordered[1].predecessor_ids


def test_routing_only_dual_discards_placeholder_lane_and_resource_order():
    target = StageInterface(
        {
            OutputSlot(0, 0): frozenset({0, 2}),
            OutputSlot(0, 1): frozenset({1, 3}),
        }
    )
    virtual = virtual_two_value_reduce()
    metadata = dict(virtual.metadata)
    metadata["routing_only"] = True

    reduced = reverse_allgather_schedule(
        replace(virtual, metadata=metadata),
        reduce_spec(),
        target,
    )

    assert {transfer.st_time for transfer in reduced.transfers} == {0.0}
    assert all(transfer.channel == 0 for transfer in reduced.transfers)
    assert all(
        not slots for slots in reduced.metadata["resource_slots"].values()
    )
    assert reduced.metadata["routing_only"] is True
    assert set(reduced.metadata["aggregate_consumptions"]) == {
        transfer.transfer_id for transfer in reduced.transfers
    }
    assert set(reduced.metadata["aggregate_states"]) >= {
        transition["produced_state_id"]
        for transition in reduced.metadata["aggregate_consumptions"].values()
    }


def test_routing_only_star_dual_builds_one_versioned_accumulator_chain():
    virtual = virtual_reduce_star(4)
    metadata = dict(virtual.metadata)
    metadata["routing_only"] = True

    reduced = reverse_allgather_schedule(
        replace(virtual, metadata=metadata),
        reduce_spec(),
        reduce_target(4),
    )
    ordered = tuple(
        sorted(reduced.transfers, key=lambda item: (item.st_time, item.transfer_id))
    )
    ordered_ids = tuple(transfer.transfer_id for transfer in ordered)

    assert tuple(
        reduced.metadata["semantic_predecessors"][transfer_id]
        for transfer_id in ordered_ids
    ) == ((), (ordered_ids[0],), (ordered_ids[1],))
    assert ordered[0].ed_time <= ordered[1].st_time
    assert ordered[1].ed_time <= ordered[2].st_time
    assert reduced.metadata["final_dependencies"] == {
        "r00000000-o00000000": (ordered_ids[-1],)
    }
    assert reduced.metadata["final_ready_times"] == {
        "r00000000-o00000000": ordered[-1].ed_time
    }

    transitions = reduced.metadata["aggregate_consumptions"]
    states = reduced.metadata["aggregate_states"]
    assert set(transitions) == set(ordered_ids)
    consumed_state_ids = []
    previous_accumulator = None
    for transfer in ordered:
        transition = transitions[transfer.transfer_id]
        consumed = tuple(transition["consumed_state_ids"])
        produced = transition["produced_state_id"]
        assert len(consumed) == 2
        assert states[consumed[0]]["contributors"] == tuple(
            sorted(transfer.member_slice_ids)
        )
        assert states[consumed[0]]["rank"] == transfer.src_rank
        assert states[consumed[1]]["rank"] == transfer.dst_rank
        assert set(states[consumed[0]]["contributors"]).isdisjoint(
            states[consumed[1]]["contributors"]
        )
        assert set(states[produced]["contributors"]) == set(
            states[consumed[0]]["contributors"]
        ) | set(states[consumed[1]]["contributors"])
        assert states[produced]["producer_id"] == transfer.transfer_id
        if previous_accumulator is not None:
            assert consumed[1] == previous_accumulator
        previous_accumulator = produced
        consumed_state_ids.extend(consumed)
    assert len(consumed_state_ids) == len(set(consumed_state_ids))
    assert states[transitions[ordered_ids[-1]]["produced_state_id"]][
        "contributors"
    ] == (0, 1, 2, 3)


def test_dual_tree_extraction_rejects_invalid_public_inputs_and_metadata():
    virtual = virtual_reduce_chain(3)

    with pytest.raises(SemanticError, match="schedule must"):
        extract_dual_trees(None, reduce_target(3))
    with pytest.raises(SemanticError, match="target_interface"):
        extract_dual_trees(virtual, None)

    metadata = dict(virtual.metadata)
    metadata["reduction_dual"] = False
    with pytest.raises(SemanticError, match="not marked"):
        extract_dual_trees(replace(virtual, metadata=metadata), reduce_target(3))

    metadata = dict(virtual.metadata)
    del metadata["tree_contributors"]
    with pytest.raises(SemanticError, match="tree_contributors"):
        extract_dual_trees(replace(virtual, metadata=metadata), reduce_target(3))

    metadata = dict(virtual.metadata)
    contributors = dict(metadata["tree_contributors"])
    contributors[virtual.transfers[0].transfer_id] = ()
    metadata["tree_contributors"] = contributors
    with pytest.raises(SemanticError, match="must not be empty"):
        extract_dual_trees(replace(virtual, metadata=metadata), reduce_target(3))


def test_dual_conversion_rejects_invalid_resource_slot_metadata():
    virtual = virtual_reduce_chain(3)
    metadata = dict(virtual.metadata)
    metadata["resource_slots"] = "invalid"

    with pytest.raises(SemanticError, match="must be a mapping"):
        reverse_allgather_schedule(
            replace(virtual, metadata=metadata),
            reduce_spec(),
            reduce_target(3),
        )

    metadata = dict(virtual.metadata)
    slots = dict(metadata["resource_slots"])
    slots[virtual.transfers[0].transfer_id] = "invalid"
    metadata["resource_slots"] = slots
    with pytest.raises(SemanticError, match="transfer resource slots"):
        reverse_allgather_schedule(
            replace(virtual, metadata=metadata),
            reduce_spec(),
            reduce_target(3),
        )
