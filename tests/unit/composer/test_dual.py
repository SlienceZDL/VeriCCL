from dataclasses import replace

import pytest

from vericcl.composer.compose import compose
from vericcl.composer.dual import reverse_allgather_schedule
from vericcl.errors import SemanticError
from vericcl.planner.dual import extract_dual_trees
from vericcl.planner.model import PlanDAG, PlanEdge, PlanNode, StageInterface
from vericcl.semantics.atom import Atom, PathStage, Schedule, Symbol, Transfer
from vericcl.semantics.collective import (
    CollectiveKind,
    CollectiveSpec,
    OutputSlot,
    required_outputs,
)
from vericcl.solver.constructive import construct_candidate
from vericcl.topology.model import LinkKey

from tests.gurobi.helpers import reduction_dual_problem
from tests.unit.composer.helpers import (
    reduce_spec,
    reduce_target,
    virtual_reduce_chain,
    virtual_reduce_star,
    virtual_two_value_reduce,
)
from tests.unit.composer.test_compose import _candidate


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


def test_star_dual_versions_one_destination_accumulator_in_order():
    reduced = reverse_allgather_schedule(
        virtual_reduce_star(3),
        reduce_spec(),
        reduce_target(3),
    )
    ordered = sorted(
        reduced.transfers,
        key=lambda transfer: (transfer.st_time, transfer.transfer_id),
    )

    assert {
        (transfer.src_rank, transfer.dst_rank)
        for transfer in reduced.transfers
    } == {(1, 0), (2, 0)}
    assert ordered[0].ed_time <= ordered[1].st_time
    assert ordered[0].transfer_id in ordered[1].predecessor_ids
    first_state = frozenset(
        reduced.metadata["tree_contributors"][ordered[0].transfer_id]
    )
    final_state = frozenset(
        reduced.metadata["tree_contributors"][ordered[1].transfer_id]
    )
    assert first_state == frozenset({0}) | ordered[0].member_slice_ids
    assert final_state == frozenset({0, 1, 2})
    assert sum(
        frozenset(contributors) == frozenset({0, 1, 2})
        for contributors in reduced.metadata["tree_contributors"].values()
    ) == 1


def test_post_reduction_send_depends_on_the_final_accumulator_version():
    contributors = frozenset({0, 1, 2})
    initial = StageInterface(
        {
            OutputSlot(rank, 0): frozenset({rank})
            for rank in range(3)
        }
    )
    reduced_value = StageInterface({OutputSlot(0, 0): contributors})
    final_values = StageInterface(
        {OutputSlot(rank, 0): contributors for rank in range(3)}
    )
    reduce_node = PlanNode(
        node_id="reduce-stage",
        stage_id=0,
        local_collective=reduce_spec(),
        communication_group=(0, 1, 2),
        logical_input=initial,
        logical_output=reduced_value,
        allowed_links=frozenset({LinkKey(1, 0), LinkKey(2, 0)}),
        shared_resource_ids=frozenset(),
        dual_of_node_id="send-stage",
    )
    send_node = PlanNode(
        node_id="send-stage",
        stage_id=1,
        local_collective=CollectiveSpec(
            kind=CollectiveKind.BROADCAST,
            datatype="float32",
            root=0,
        ),
        communication_group=(0, 1, 2),
        logical_input=reduced_value,
        logical_output=final_values,
        allowed_links=frozenset({LinkKey(0, 1), LinkKey(0, 2)}),
        shared_resource_ids=frozenset(),
    )
    collective = CollectiveSpec(
        kind=CollectiveKind.ALL_REDUCE,
        datatype="float32",
        reduction_op="sum",
    )
    plan = PlanDAG(
        collective=collective,
        rank_count=3,
        slice_count=1,
        initial_inputs=initial,
        nodes=(reduce_node, send_node),
        edges=(
            PlanEdge(
                producer_id=reduce_node.node_id,
                consumer_id=send_node.node_id,
                interface=reduced_value,
            ),
        ),
        final_outputs=StageInterface(required_outputs(collective, 3, 1)),
    )
    transfers = []
    path_roots = {}
    semantic_contributors = {}
    tree_contributors = {}
    semantic_predecessors = {}
    resource_slots = {}
    for destination in (1, 2):
        transfer_id = "post-reduce-send-{}".format(destination)
        atoms = tuple(
            Atom(
                slice_id=member,
                slice_size_bytes=1024,
                path=(PathStage(1, "SEND", (Symbol(0, destination, 0.0),)),),
                st_time=0.0,
                ed_time=1.0,
            )
            for member in sorted(contributors)
        )
        transfers.append(
            Transfer(
                transfer_id=transfer_id,
                kind="SEND",
                src_rank=0,
                dst_rank=destination,
                channel=0,
                stage_id=1,
                member_slice_ids=contributors,
                atoms=atoms,
                st_time=0.0,
                ed_time=1.0,
                predecessor_ids=frozenset(),
            )
        )
        path_roots[transfer_id] = 0
        semantic_contributors[transfer_id] = tuple(sorted(contributors))
        tree_contributors[transfer_id] = tuple(sorted(contributors))
        semantic_predecessors[transfer_id] = ()
        resource_slots[transfer_id] = {}
    send_schedule = Schedule(
        schedule_id="post-reduce-send",
        transfers=tuple(transfers),
        final_state_ids=(),
        rank_count=3,
        slice_count=1,
        slice_size_bytes=1024,
        metadata={
            "path_scope": "stage_suffix",
            "path_roots": path_roots,
            "reduction_dual": False,
            "semantic_contributors": semantic_contributors,
            "tree_contributors": tree_contributors,
            "semantic_predecessors": semantic_predecessors,
            "resource_slots": resource_slots,
        },
    )

    composed = compose(
        plan,
        {
            reduce_node.node_id: _candidate(
                reduce_node.node_id,
                virtual_reduce_star(3),
            ),
            send_node.node_id: _candidate(send_node.node_id, send_schedule),
        },
    )
    reductions = sorted(
        (transfer for transfer in composed.transfers if transfer.kind == "REDUCE"),
        key=lambda transfer: (transfer.st_time, transfer.transfer_id),
    )
    sends = tuple(
        transfer for transfer in composed.transfers if transfer.kind == "SEND"
    )

    assert len(reductions) == 2
    assert reductions[0].transfer_id in reductions[1].predecessor_ids
    assert all(
        reductions[1].transfer_id in transfer.predecessor_ids
        for transfer in sends
    )
    assert all(
        reductions[0].transfer_id not in transfer.predecessor_ids
        for transfer in sends
    )


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
