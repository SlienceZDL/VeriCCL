from dataclasses import replace
from pathlib import Path

import pytest

from vericcl.composer.compose import compose
from vericcl.errors import SemanticError
from vericcl.input.loader import resolve_inputs
from vericcl.input.models import AtomConstraints, ForbiddenTransfer, ObjectiveMode
from vericcl.planner.model import (
    PlanDAG,
    PlanEdge,
    PlanNode,
    PlanningMode,
    StageInterface,
)
from vericcl.semantics.collective import (
    CollectiveKind,
    CollectiveSpec,
    OutputSlot,
    required_outputs,
)
from vericcl.solver import (
    InstantiationFailure,
    InstantiationResult,
    instantiate_route_patterns,
)
from vericcl.solver.demands import build_solver_problem
from vericcl.solver.instantiate import (
    _MemberInstantiationError,
    _validate_mapped_path,
)
from vericcl.solver.model import SolveCandidate, SolveStatus, SolverMetrics
from vericcl.solver.routing import RoutePattern, RoutingModelStats
from vericcl.solver.scheduling import RoutedOperation, RoutedTree
from vericcl.solver.templates import build_solver_templates, split_routing_units
from vericcl.topology.model import (
    DirectedLink,
    LinkKey,
    PerformanceCurve,
    SharedResource,
    Topology,
)
from vericcl.verification.model import ValidationStatus
from vericcl.verification.semantics import verify_schedule_semantics
from vericcl.xml import (
    AggregateValue,
    EndpointType,
    build_buffer_plan,
    build_transfer_dag,
    lower_endpoints,
)


pytestmark = pytest.mark.phase03


EXAMPLES = Path(__file__).parents[3] / "vericcl" / "examples"


def _inputs(kind, rank_count, slice_count, *, root=None):
    base = resolve_inputs(
        EXAMPLES / "topo" / "two_rank.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    reduced = kind in {
        CollectiveKind.REDUCE,
        CollectiveKind.ALL_REDUCE,
        CollectiveKind.REDUCE_SCATTER,
    }
    return replace(
        base,
        collective=CollectiveSpec(
            kind=kind,
            datatype="float32",
            reduction_op="sum" if reduced else None,
            root=root,
        ),
        hyperparameters=replace(
            base.hyperparameters,
            total_size_bytes=slice_count * 1024,
            slice_size_bytes=1024,
        ),
        rank_count=rank_count,
        strategies=replace(
            base.strategies,
            hierarchy=False,
            shortest_paths=True,
            batching=False,
        ),
        atom_constraints=AtomConstraints(
            stage_num=None,
            forbidden_transfers=(),
        ),
    )


def _topology(rank_count, links, *, shared=False):
    keys = tuple(LinkKey(src, dst) for src, dst in links)
    curve = PerformanceCurve(1.0, 2.0, {})
    resources = {}
    resource_ids = ()
    if shared:
        resource = SharedResource(
            resource_id="fabric",
            member_links=keys,
            max_channels=2,
            performance=curve,
        )
        resources[resource.resource_id] = resource
        resource_ids = (resource.resource_id,)
    return Topology(
        rank_count=rank_count,
        links={
            key: DirectedLink(
                key=key,
                max_channels=2,
                performance=curve,
                resource_ids=resource_ids,
            )
            for key in keys
        },
        shared_resources=resources,
        node_membership={rank: 0 for rank in range(rank_count)},
        gateways=frozenset(),
        warnings=(),
    )


def _initial_interface(rank_count, slice_count):
    return StageInterface(
        {
            OutputSlot(rank, logical): frozenset(
                {rank * slice_count + logical}
            )
            for rank in range(rank_count)
            for logical in range(slice_count)
        }
    )


def _metrics():
    return SolverMetrics(
        status=SolveStatus.OPTIMAL,
        objective_values=(1.0, 1.0, 1.0),
        best_bound=1.0,
        mip_gap=0.0,
        within_requested_gap=True,
        solve_time_s=0.0,
        model_count=1,
        operation_count=1,
        hop_count=1,
        makespan_us=1.0,
        maximum_normalized_resource_load=1.0,
        solver_name="test",
        solver_version="test",
        solver_seed=0,
        thread_count=1,
        termination_reason="optimal",
    )


def _patterns(templates, channel_count=2):
    result = {}
    for template in templates:
        paths = tuple(
            (
                demand.demand_id,
                tuple(zip(path, path[1:])),
            )
            for demand in template.representative.demands
            for path in (demand.candidate_paths[0],)
        )
        selected = tuple(
            sorted({edge for _, path in paths for edge in path})
        )
        result[template.template_id] = RoutePattern(
            template_id=template.template_id,
            channel_count=channel_count,
            objective_mode=ObjectiveMode.LATENCY,
            selected_edges=selected,
            member_paths=paths,
            metrics=_metrics(),
            model_stats=RoutingModelStats(1, 1, 0, 0.0, 0.0),
        )
    return result


def _broadcast_fixture():
    rank_count = 3
    slice_count = 2
    inputs = _inputs(
        CollectiveKind.BROADCAST,
        rank_count,
        slice_count,
        root=0,
    )
    topology = _topology(
        rank_count,
        ((0, 1), (1, 2)),
        shared=True,
    )
    outputs = required_outputs(inputs.collective, rank_count, slice_count)
    node = PlanNode(
        node_id="broadcast-chain",
        stage_id=0,
        local_collective=inputs.collective,
        communication_group=tuple(range(rank_count)),
        logical_input=StageInterface(
            {
                OutputSlot(0, logical): frozenset({logical})
                for logical in range(slice_count)
            }
        ),
        logical_output=StageInterface(outputs),
        allowed_links=frozenset(topology.links),
        shared_resource_ids=frozenset(topology.shared_resources),
    )
    plan = PlanDAG(
        collective=inputs.collective,
        rank_count=rank_count,
        slice_count=slice_count,
        initial_inputs=_initial_interface(rank_count, slice_count),
        nodes=(node,),
        edges=(),
        final_outputs=StageInterface(outputs),
        planning_mode=PlanningMode.DIRECT,
    )
    problem = build_solver_problem(node, inputs, topology)
    templates = build_solver_templates((problem,), plan.planning_mode)
    return inputs, topology, plan, templates, _patterns(templates)


def _allgather_fixture():
    rank_count = 3
    slice_count = 2
    inputs = _inputs(CollectiveKind.ALL_GATHER, rank_count, slice_count)
    topology = _topology(
        rank_count,
        tuple(
            (src, dst)
            for src in range(rank_count)
            for dst in range(rank_count)
            if src != dst
        ),
    )
    initial = _initial_interface(rank_count, slice_count)
    outputs = required_outputs(inputs.collective, rank_count, slice_count)
    node = PlanNode(
        node_id="allgather",
        stage_id=0,
        local_collective=inputs.collective,
        communication_group=tuple(range(rank_count)),
        logical_input=initial,
        logical_output=StageInterface(outputs),
        allowed_links=frozenset(topology.links),
        shared_resource_ids=frozenset(),
    )
    plan = PlanDAG(
        collective=inputs.collective,
        rank_count=rank_count,
        slice_count=slice_count,
        initial_inputs=initial,
        nodes=(node,),
        edges=(),
        final_outputs=StageInterface(outputs),
        planning_mode=PlanningMode.DIRECT,
    )
    problem = build_solver_problem(node, inputs, topology)
    templates = build_solver_templates((problem,), plan.planning_mode)
    return inputs, topology, plan, templates, _patterns(templates)


def _allreduce_fixture():
    rank_count = 4
    slice_count = 1
    inputs = _inputs(CollectiveKind.ALL_REDUCE, rank_count, slice_count)
    topology = _topology(
        rank_count,
        tuple(
            (src, dst)
            for src in range(rank_count)
            for dst in range(rank_count)
            if src != dst
        ),
    )
    aggregate = frozenset(range(rank_count))
    reduce_node = PlanNode(
        node_id="reduce-star",
        stage_id=0,
        local_collective=CollectiveSpec(
            kind=CollectiveKind.REDUCE,
            datatype="float32",
            reduction_op="sum",
            root=0,
        ),
        communication_group=tuple(range(rank_count)),
        logical_input=StageInterface(
            {
                OutputSlot(rank, 0): frozenset({rank})
                for rank in range(rank_count)
            }
        ),
        logical_output=StageInterface({OutputSlot(0, 0): aggregate}),
        allowed_links=frozenset(topology.links),
        shared_resource_ids=frozenset(),
        dual_of_node_id="reduce-star-virtual",
    )
    broadcast_node = PlanNode(
        node_id="broadcast-aggregate",
        stage_id=1,
        local_collective=CollectiveSpec(
            kind=CollectiveKind.BROADCAST,
            datatype="float32",
            root=0,
        ),
        communication_group=tuple(range(rank_count)),
        logical_input=StageInterface({OutputSlot(0, 0): aggregate}),
        logical_output=StageInterface(
            {OutputSlot(rank, 0): aggregate for rank in range(rank_count)}
        ),
        allowed_links=frozenset(topology.links),
        shared_resource_ids=frozenset(),
    )
    outputs = required_outputs(inputs.collective, rank_count, slice_count)
    plan = PlanDAG(
        collective=inputs.collective,
        rank_count=rank_count,
        slice_count=slice_count,
        initial_inputs=_initial_interface(rank_count, slice_count),
        nodes=(reduce_node, broadcast_node),
        edges=(
            PlanEdge(
                reduce_node.node_id,
                broadcast_node.node_id,
                StageInterface({OutputSlot(0, 0): aggregate}),
            ),
        ),
        final_outputs=StageInterface(outputs),
        planning_mode=PlanningMode.DIRECT,
    )
    problems = tuple(
        build_solver_problem(node, inputs, topology) for node in plan.nodes
    )
    templates = build_solver_templates(problems, plan.planning_mode)
    return inputs, topology, plan, templates, _patterns(templates)


def test_logical_position_instantiation_rebuilds_real_paths_and_provisional_policy():
    inputs, topology, plan, templates, patterns = _broadcast_fixture()

    first = instantiate_route_patterns(
        plan,
        templates,
        patterns,
        inputs,
        topology,
    )
    second = instantiate_route_patterns(
        plan,
        templates,
        patterns,
        inputs,
        topology,
    )

    assert first == second
    assert isinstance(first, InstantiationResult)
    assert first.failures == ()
    schedule = first.node_schedules["broadcast-chain"]
    assert schedule.metadata["routing_only"] is True
    assert all(transfer.channel == 0 for transfer in schedule.transfers)
    assert all(
        not slots for slots in schedule.metadata["resource_slots"].values()
    )
    leaf = next(
        transfer
        for transfer in schedule.transfers
        if transfer.dst_rank == 2 and transfer.member_slice_ids == {1}
    )
    assert leaf.st_time == 3.0
    assert leaf.ed_time == 6.0
    assert [
        (symbol.src_rank, symbol.dst_rank)
        for symbol in leaf.atoms[0].path[0].symbols
    ] == [(0, 1), (1, 2)]
    assert leaf.atoms[0].path[0].symbols[0].src_rank == 0
    assert leaf.transfer_id in schedule.metadata["semantic_predecessors"]
    assert schedule.metadata["final_outputs"]["r00000002-o00000001"] == (1,)


def test_allgather_preserves_local_values_after_output_offset_remapping():
    inputs, topology, plan, templates, patterns = _allgather_fixture()

    result = instantiate_route_patterns(
        plan,
        templates,
        patterns,
        inputs,
        topology,
    )

    assert result.failures == ()
    schedule = result.node_schedules["allgather"]
    expected = {
        "r{:08d}-o{:08d}".format(slot.rank, slot.offset): tuple(
            sorted(contributors)
        )
        for slot, contributors in plan.final_outputs.values.items()
    }
    assert schedule.metadata["final_outputs"] == expected
    assert len(schedule.final_state_ids) == len(expected)
    assert verify_schedule_semantics(schedule, inputs).status is ValidationStatus.VALID


def test_one_forbidden_member_falls_back_without_partial_schedule():
    inputs, topology, plan, templates, patterns = _broadcast_fixture()
    forbidden_inputs = replace(
        inputs,
        atom_constraints=AtomConstraints(
            stage_num=None,
            forbidden_transfers=(ForbiddenTransfer(1, 1, 2, 0),),
        ),
    )

    result = instantiate_route_patterns(
        plan,
        templates,
        patterns,
        forbidden_inputs,
        topology,
    )

    assert len(result.failures) == 1
    assert isinstance(result.failures[0], InstantiationFailure)
    assert result.failures[0].node_id == "broadcast-chain"
    assert result.failures[0].reason == "mapped_route_hits_forbidden_transfer"
    assert all(
        1 not in transfer.member_slice_ids
        for transfer in result.node_schedules["broadcast-chain"].transfers
    )
    assert {
        member
        for transfer in result.node_schedules["broadcast-chain"].transfers
        for member in transfer.member_slice_ids
    } == {0}


def test_reduction_dual_rebuilds_join_and_downstream_dependencies():
    inputs, topology, plan, templates, patterns = _allreduce_fixture()

    result = instantiate_route_patterns(
        plan,
        templates,
        patterns,
        inputs,
        topology,
    )

    assert result.failures == ()
    reduce_schedule = result.node_schedules["reduce-star"]
    reductions = tuple(reduce_schedule.transfers)
    assert len(reductions) == 3
    assert all(transfer.kind == "REDUCE" for transfer in reductions)
    assert {
        member
        for transfer in reductions
        for member in transfer.member_slice_ids
    } == {1, 2, 3}
    assert sum(len(transfer.member_slice_ids) for transfer in reductions) == 3
    reduce_ids = {transfer.transfer_id for transfer in reductions}
    assert set(
        reduce_schedule.metadata["final_dependencies"][
            "r00000000-o00000000"
        ]
    ) == reduce_ids
    assert set(
        reduce_schedule.metadata["aggregate_consumptions"][
            "r00000000-o00000000"
        ]
    ) == reduce_ids

    candidates = {
        node_id: SolveCandidate(
            candidate_id="candidate-{}".format(node_id),
            node_schedules={node_id: schedule},
            objective_mode=ObjectiveMode.LATENCY,
            channel_count=2,
            metrics=_metrics(),
            selected_best=False,
            proven_optimal=False,
            search_space_restricted=False,
            restrictions=(),
            parent_candidate_id=None,
        )
        for node_id, schedule in result.node_schedules.items()
    }
    composed = compose(plan, candidates)
    sends = [
        transfer for transfer in composed.transfers if transfer.stage_id == 1
    ]
    semantic = composed.metadata["semantic_predecessors"]
    assert sends
    assert all(reduce_ids <= set(semantic[transfer.transfer_id]) for transfer in sends)
    assert verify_schedule_semantics(composed, inputs).status is ValidationStatus.VALID

    buffers = build_buffer_plan(composed, inputs)
    endpoints = lower_endpoints(composed, buffers)
    transfer_dag = build_transfer_dag(endpoints, composed, buffers)
    final_reductions = {
        transfer_id
        for transfer_id, value in buffers.transfer_output_values.items()
        if transfer_id in reduce_ids
        and isinstance(value, AggregateValue)
        and value.contributors == frozenset(range(4))
    }
    assert len(final_reductions) == 1
    final_reduction = next(iter(final_reductions))
    assert any(
        endpoint.transfer_id == final_reduction
        and endpoint.xml_type is EndpointType.RECV_REDUCE_COPY
        for endpoint in endpoints.endpoints
    )
    assert all(
        final_reduction in transfer_dag.predecessors[transfer.transfer_id]
        for transfer in sends
    )


def test_instantiation_models_reject_invalid_values():
    inputs, topology, plan, templates, patterns = _broadcast_fixture()
    schedule = instantiate_route_patterns(
        plan,
        templates,
        patterns,
        inputs,
        topology,
    ).node_schedules["broadcast-chain"]

    invalid = (
        lambda: InstantiationFailure("", "node", "reason"),
        lambda: InstantiationResult(None, ()),
        lambda: InstantiationResult({"": schedule}, ()),
        lambda: InstantiationResult({}, (object(),)),
    )
    for constructor in invalid:
        with pytest.raises(SemanticError):
            constructor()


def test_instantiate_rejects_invalid_public_inputs():
    inputs, topology, plan, templates, patterns = _broadcast_fixture()
    pattern = next(iter(patterns.values()))
    invalid_arguments = (
        (None, templates, patterns, inputs, topology),
        (plan, templates, patterns, None, topology),
        (plan, templates, patterns, inputs, None),
        (plan, templates, patterns, replace(inputs, rank_count=4), topology),
        (plan, None, patterns, inputs, topology),
        (plan, (object(),), patterns, inputs, topology),
        (plan, templates, {1: pattern}, inputs, topology),
        (plan, (templates[0], templates[0]), patterns, inputs, topology),
        (plan, templates, {"unknown-template": pattern}, inputs, topology),
    )

    for arguments in invalid_arguments:
        with pytest.raises(SemanticError):
            instantiate_route_patterns(*arguments)


def test_missing_pattern_and_routing_unit_produce_stable_failures():
    inputs, topology, plan, templates, patterns = _broadcast_fixture()

    missing_patterns = instantiate_route_patterns(
        plan,
        templates,
        {},
        inputs,
        topology,
    )
    assert {
        failure.reason for failure in missing_patterns.failures
    } == {"route_pattern_missing"}
    assert missing_patterns.node_schedules["broadcast-chain"].transfers == ()

    template = templates[0]
    missing_member = replace(
        template.members[0],
        unit_id="missing-routing-unit",
    )
    extended = replace(
        template,
        members=template.members + (missing_member,),
    )
    missing_unit = instantiate_route_patterns(
        plan,
        (extended,),
        patterns,
        inputs,
        topology,
    )
    assert any(
        failure.unit_id == "missing-routing-unit"
        and failure.reason == "routing_unit_missing"
        for failure in missing_unit.failures
    )


def test_mapped_path_revalidation_rejects_each_domain_violation():
    inputs, topology, plan, _, _ = _broadcast_fixture()
    node = plan.nodes[0]
    demand = next(
        item
        for item in build_solver_problem(node, inputs, topology).demands
        if item.required_leaf_rank == 2
    )
    valid_path = (LinkKey(0, 1), LinkKey(1, 2))
    cases = (
        (
            demand,
            (LinkKey(1, 2),),
            node,
            "mapped_route_has_invalid_geometry",
        ),
        (
            demand,
            (LinkKey(0, 2),),
            node,
            "mapped_route_missing_topology_edge",
        ),
        (
            demand,
            valid_path,
            replace(node, allowed_links=frozenset({LinkKey(0, 1)})),
            "mapped_route_outside_plan_node_domain",
        ),
        (
            replace(
                demand,
                allowed_links=frozenset(),
                legal_links=frozenset(),
                candidate_paths=(),
            ),
            valid_path,
            node,
            "mapped_route_outside_demand_domain",
        ),
        (
            replace(demand, legal_links=frozenset(), candidate_paths=()),
            valid_path,
            node,
            "mapped_route_outside_legal_domain",
        ),
        (
            demand,
            valid_path,
            replace(node, shared_resource_ids=frozenset()),
            "mapped_route_has_invalid_resource_membership",
        ),
        (
            replace(demand, candidate_paths=()),
            valid_path,
            node,
            "mapped_route_outside_candidate_path_domain",
        ),
    )

    for current_demand, path, current_node, reason in cases:
        with pytest.raises(_MemberInstantiationError, match=reason):
            _validate_mapped_path(
                current_demand,
                path,
                current_node,
                topology,
            )


def test_shared_route_materializer_models_reject_invalid_values():
    inputs, topology, plan, _, _ = _broadcast_fixture()
    problem = build_solver_problem(plan.nodes[0], inputs, topology)
    unit = split_routing_units(problem)[0]
    demand = unit.demands[0]
    path = tuple(
        LinkKey(src, dst)
        for src, dst in zip(
            demand.candidate_paths[0],
            demand.candidate_paths[0][1:],
        )
    )
    tree_arguments = {
        "route_id": "route",
        "root_rank": demand.root_rank,
        "logical_position": demand.logical_position,
        "contributors": demand.contributors,
        "reduction_dual": demand.reduction_dual,
        "demands": (demand,),
        "selected_paths": ((demand.demand_id, path),),
    }
    invalid_trees = (
        {"route_id": ""},
        {"root_rank": True},
        {"root_rank": -1},
        {"logical_position": -1},
        {"contributors": frozenset()},
        {"reduction_dual": 1},
        {"demands": ()},
        {"demands": (demand, demand)},
        {"selected_paths": (("invalid",),)},
        {"selected_paths": ()},
        {"selected_paths": ((demand.demand_id, ()),)},
    )
    for changes in invalid_trees:
        with pytest.raises(SemanticError):
            RoutedTree(**{**tree_arguments, **changes})

    operation_arguments = {
        "route_id": "route",
        "link": LinkKey(0, 1),
        "channel": 0,
        "st_time": 0.0,
        "ed_time": 1.0,
        "resource_slots": (),
    }
    invalid_operations = (
        {"route_id": ""},
        {"link": object()},
        {"channel": -1},
        {"st_time": float("nan")},
        {"st_time": 2.0},
        {"resource_slots": (object(),)},
        {"resource_slots": (("", 0),)},
        {"resource_slots": (("nic", 0), ("nic", 1))},
    )
    for changes in invalid_operations:
        with pytest.raises(SemanticError):
            RoutedOperation(**{**operation_arguments, **changes})
