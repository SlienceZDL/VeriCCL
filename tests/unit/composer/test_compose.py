from dataclasses import replace

import pytest

from vericcl.composer.compose import (
    _output_path_transfers,
    compose,
    compose_routes,
)
from vericcl.composer.dual import reverse_allgather_schedule
from vericcl.errors import SemanticError
from vericcl.input.models import ObjectiveMode
from vericcl.planner.model import (
    PlanDAG,
    PlanEdge,
    PlanNode,
    StageInterface,
)
from vericcl.semantics.atom import Atom, PathStage, Schedule, Symbol, Transfer
from vericcl.semantics.collective import (
    CollectiveKind,
    CollectiveSpec,
    OutputSlot,
    required_outputs,
)
from vericcl.solver.model import SolveCandidate, SolveStatus, SolverMetrics
from vericcl.topology.model import LinkKey

from tests.unit.verification.simulator_helpers import (
    curve,
    simulation_topology,
)

from tests.unit.composer.helpers import (
    reduce_spec,
    reduce_target,
    virtual_reduce_chain,
)


pytestmark = pytest.mark.phase03


def _interface(entries):
    return StageInterface(
        {
            OutputSlot(rank, offset): frozenset(contributors)
            for rank, offset, contributors in entries
        }
    )


def _pipeline_plan():
    collective = CollectiveSpec(
        kind=CollectiveKind.BROADCAST,
        datatype="float32",
        root=0,
    )
    initial = _interface(
        (rank, offset, {rank * 2 + offset})
        for rank in range(3)
        for offset in range(2)
    )
    root_values = [(0, offset, {offset}) for offset in range(2)]
    middle_values = [(1, offset, {offset}) for offset in range(2)]
    leaf_values = [(2, offset, {offset}) for offset in range(2)]
    first = PlanNode(
        node_id="pipeline-first",
        stage_id=0,
        local_collective=CollectiveSpec(
            kind=CollectiveKind.BROADCAST,
            datatype="float32",
            root=0,
        ),
        communication_group=(0, 1),
        logical_input=_interface(root_values),
        logical_output=_interface(root_values + middle_values),
        allowed_links=frozenset({LinkKey(0, 1)}),
        shared_resource_ids=frozenset(),
    )
    second = PlanNode(
        node_id="pipeline-second",
        stage_id=1,
        local_collective=CollectiveSpec(
            kind=CollectiveKind.BROADCAST,
            datatype="float32",
            root=1,
        ),
        communication_group=(1, 2),
        logical_input=_interface(middle_values),
        logical_output=_interface(middle_values + leaf_values),
        allowed_links=frozenset({LinkKey(1, 2)}),
        shared_resource_ids=frozenset(),
    )
    return PlanDAG(
        collective=collective,
        rank_count=3,
        slice_count=2,
        initial_inputs=initial,
        nodes=(first, second),
        edges=(
            PlanEdge(
                producer_id=first.node_id,
                consumer_id=second.node_id,
                interface=_interface(middle_values),
            ),
        ),
        final_outputs=StageInterface(required_outputs(collective, 3, 2)),
    )


def _node_schedule(node_id, stage_id, src_rank, dst_rank):
    transfers = []
    path_roots = {}
    contributors = {}
    tree_contributors = {}
    semantic_predecessors = {}
    resource_slots = {}
    for slice_id in range(2):
        transfer_id = "{}-t{}".format(node_id, slice_id)
        st_time = float(2 * slice_id)
        ed_time = st_time + 2.0
        atom = Atom(
            slice_id=slice_id,
            slice_size_bytes=1024,
            path=(
                PathStage(
                    stage_id,
                    "SEND",
                    (Symbol(src_rank, dst_rank, 0.0),),
                ),
            ),
            st_time=st_time,
            ed_time=ed_time,
        )
        transfers.append(
            Transfer(
                transfer_id=transfer_id,
                kind="SEND",
                src_rank=src_rank,
                dst_rank=dst_rank,
                channel=0,
                stage_id=stage_id,
                member_slice_ids=frozenset({slice_id}),
                atoms=(atom,),
                st_time=st_time,
                ed_time=ed_time,
                predecessor_ids=(
                    frozenset({"{}-t0".format(node_id)})
                    if slice_id
                    else frozenset()
                ),
            )
        )
        path_roots[transfer_id] = src_rank
        contributors[transfer_id] = (slice_id,)
        tree_contributors[transfer_id] = (slice_id,)
        semantic_predecessors[transfer_id] = ()
        resource_slots[transfer_id] = {}
    return Schedule(
        schedule_id="{}-schedule".format(node_id),
        transfers=tuple(transfers),
        final_state_ids=(),
        rank_count=3,
        slice_count=2,
        slice_size_bytes=1024,
        metadata={
            "path_scope": "stage_suffix",
            "path_roots": path_roots,
            "semantic_contributors": contributors,
            "tree_contributors": tree_contributors,
            "semantic_predecessors": semantic_predecessors,
            "resource_slots": resource_slots,
            "reduction_dual": False,
        },
    )


def _candidate(node_id, schedule):
    metrics = SolverMetrics(
        status=SolveStatus.FEASIBLE,
        objective_values=(4.0, 2.0, 2.0),
        best_bound=0.0,
        mip_gap=0.0,
        within_requested_gap=True,
        solve_time_s=0.0,
        model_count=1,
        operation_count=2,
        hop_count=2,
        makespan_us=4.0,
        maximum_normalized_resource_load=4.0,
        solver_name="test",
        solver_version="1",
        solver_seed=0,
        thread_count=1,
        termination_reason="complete",
    )
    return SolveCandidate(
        candidate_id="{}-candidate".format(node_id),
        node_schedules={node_id: schedule},
        objective_mode=ObjectiveMode.LATENCY,
        channel_count=1,
        metrics=metrics,
        selected_best=False,
        proven_optimal=False,
        search_space_restricted=False,
        restrictions=(),
        parent_candidate_id=None,
    )


def _pipeline_candidates():
    return {
        "pipeline-first": _candidate(
            "pipeline-first",
            _node_schedule("pipeline-first", 0, 0, 1),
        ),
        "pipeline-second": _candidate(
            "pipeline-second",
            _node_schedule("pipeline-second", 1, 1, 2),
        ),
    }


def _pipeline_route_schedules(plan):
    nodes = {node.node_id: node for node in plan.nodes}
    schedules = {}
    for node_id, candidate in _pipeline_candidates().items():
        schedule = candidate.node_schedules[node_id]
        metadata = dict(schedule.metadata)
        final_outputs = {}
        final_dependencies = {}
        for slot, contributors in nodes[node_id].logical_output.values.items():
            key = "r{:08d}-o{:08d}".format(slot.rank, slot.offset)
            final_outputs[key] = tuple(sorted(contributors))
            final_dependencies[key] = tuple(
                sorted(
                    transfer.transfer_id
                    for transfer in schedule.transfers
                    if transfer.dst_rank == slot.rank
                    and frozenset(
                        metadata["tree_contributors"][transfer.transfer_id]
                    )
                    == contributors
                )
            )
        metadata.update(
            {
                "routing_only": True,
                "final_outputs": final_outputs,
                "final_dependencies": final_dependencies,
            }
        )
        schedules[node_id] = replace(schedule, metadata=metadata)
    return schedules


def test_composer_pipelines_ready_slice_without_stage_barrier():
    schedule = compose(_pipeline_plan(), _pipeline_candidates())
    by_id = {transfer.transfer_id: transfer for transfer in schedule.transfers}

    assert (
        by_id["pipeline-second-t0"].st_time
        < by_id["pipeline-first-t1"].ed_time
    )
    assert by_id["pipeline-second-t0"].st_time == 2.0
    assert by_id["pipeline-first-t1"].ed_time == 4.0
    atom = by_id["pipeline-second-t0"].atoms[0]
    assert [stage.stage_id for stage in atom.path] == [0, 1]
    assert [
        symbol.ready_time
        for stage in atom.path
        for symbol in stage.symbols
    ] == [0.0, 2.0]
    assert schedule.metadata["path_scope"] == "global"


def test_compose_routes_assigns_resources_after_cross_node_semantics():
    plan = _pipeline_plan()
    schedules = _pipeline_route_schedules(plan)
    topology = simulation_topology(
        3,
        {
            (0, 1): curve(),
            (1, 2): curve(),
        },
        max_channels=1,
    )

    schedule = compose_routes(
        plan,
        schedules,
        topology,
        1,
    )
    by_id = {transfer.transfer_id: transfer for transfer in schedule.transfers}

    assert by_id["pipeline-second-t0"].st_time == 2.0
    assert by_id["pipeline-second-t0"].st_time < by_id[
        "pipeline-first-t1"
    ].ed_time
    assert schedule.metadata["global_resources_assigned"] is True
    assert "routing_only" not in schedule.metadata


@pytest.mark.parametrize("channel_count", (0, -1, True, 1.0, 33))
def test_compose_routes_rejects_invalid_channel_count(channel_count):
    plan = _pipeline_plan()
    topology = simulation_topology(
        3,
        {
            (0, 1): curve(),
            (1, 2): curve(),
        },
        max_channels=1,
    )

    with pytest.raises(SemanticError, match="channel_count"):
        compose_routes(
            plan,
            _pipeline_route_schedules(plan),
            topology,
            channel_count,
        )


def test_compose_routes_accepts_maximum_channel_count():
    plan = _pipeline_plan()
    topology = simulation_topology(
        3,
        {
            (0, 1): curve(),
            (1, 2): curve(),
        },
        max_channels=1,
    )

    schedule = compose_routes(
        plan,
        _pipeline_route_schedules(plan),
        topology,
        32,
    )

    assert schedule.metadata["channel_count"] == 32


def test_composer_requires_one_complete_candidate_per_plan_node():
    candidates = _pipeline_candidates()
    del candidates["pipeline-second"]

    with pytest.raises(SemanticError, match="candidate"):
        compose(_pipeline_plan(), candidates)


def test_composer_supports_single_rank_collective_without_transfers():
    collective = CollectiveSpec(
        kind=CollectiveKind.BROADCAST,
        datatype="float32",
        root=0,
    )
    interface = _interface(((0, 0, {0}),))
    node = PlanNode(
        node_id="singleton",
        stage_id=0,
        local_collective=collective,
        communication_group=(0,),
        logical_input=interface,
        logical_output=interface,
        allowed_links=frozenset(),
        shared_resource_ids=frozenset(),
    )
    plan = PlanDAG(
        collective=collective,
        rank_count=1,
        slice_count=1,
        initial_inputs=interface,
        nodes=(node,),
        edges=(),
        final_outputs=interface,
    )
    local = Schedule(
        schedule_id="singleton-schedule",
        transfers=(),
        final_state_ids=(),
        rank_count=1,
        slice_count=1,
        slice_size_bytes=1024,
        metadata={
            "path_scope": "stage_suffix",
            "path_roots": {},
            "semantic_contributors": {},
            "tree_contributors": {},
            "semantic_predecessors": {},
            "resource_slots": {},
            "reduction_dual": False,
        },
    )

    result = compose(plan, {"singleton": _candidate("singleton", local)})

    assert result.transfers == ()
    assert result.slice_size_bytes == 1024
    assert result.metadata["final_outputs"] == {
        "r00000000-o00000000": (0,)
    }
    assert result.metadata["final_dependencies"] == {
        "r00000000-o00000000": ()
    }


def test_composer_rejects_invalid_stage_resource_slot_metadata():
    candidates = _pipeline_candidates()
    candidate = candidates["pipeline-first"]
    schedule = candidate.node_schedules["pipeline-first"]
    metadata = dict(schedule.metadata)
    metadata["resource_slots"] = "invalid"
    schedule = replace(schedule, metadata=metadata)
    candidates["pipeline-first"] = replace(
        candidate,
        node_schedules={"pipeline-first": schedule},
    )

    with pytest.raises(SemanticError, match="resource_slots"):
        compose(_pipeline_plan(), candidates)


def test_routing_output_paths_prefer_the_terminal_aggregate_transfer():
    virtual = virtual_reduce_chain(3)
    identifiers = {
        "virtual-t0000": "z-terminal",
        "virtual-t0001": "a-child",
    }
    transfers = tuple(
        replace(
            transfer,
            transfer_id=identifiers[transfer.transfer_id],
            predecessor_ids=frozenset(
                identifiers[predecessor]
                for predecessor in transfer.predecessor_ids
            ),
        )
        for transfer in virtual.transfers
    )
    metadata = dict(virtual.metadata)
    for field in (
        "path_roots",
        "semantic_contributors",
        "tree_contributors",
        "resource_slots",
    ):
        metadata[field] = {
            identifiers[transfer_id]: value
            for transfer_id, value in metadata[field].items()
        }
    metadata["routing_only"] = True
    reduced = reverse_allgather_schedule(
        replace(virtual, transfers=transfers, metadata=metadata),
        reduce_spec(),
        reduce_target(3),
    )
    terminal = next(
        transfer for transfer in reduced.transfers if transfer.dst_rank == 0
    )

    ordered = _output_path_transfers(reduced, (terminal,))
    member_source = next(
        transfer
        for transfer in ordered
        if any(atom.slice_id == 2 for atom in transfer.atoms)
    )

    assert member_source.transfer_id == terminal.transfer_id
