from dataclasses import replace

import pytest

from vericcl.composer.dual import reverse_allgather_schedule
from vericcl.errors import SemanticError
from vericcl.planner.dual import extract_dual_trees
from vericcl.planner.model import StageInterface
from vericcl.semantics.atom import Schedule
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
    assert (
        reduced.metadata["aggregate_consumptions"]
        == reduced.metadata["final_dependencies"]
    )


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
