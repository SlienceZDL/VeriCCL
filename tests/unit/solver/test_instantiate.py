from dataclasses import replace

import pytest

from vericcl.composer.dual import reverse_allgather_schedule
from vericcl.input.models import ForbiddenTransfer, ObjectiveMode
from vericcl.planner.model import PlanningMode
from vericcl.semantics.slice import source_rank
from vericcl.solver.instantiate import instantiate_route_patterns
from vericcl.solver.routing import RoutePattern, RoutingModelStats
from vericcl.solver.templates import build_solver_templates, split_routing_units
from vericcl.topology.model import LinkKey

from tests.gurobi.helpers import reduction_dual_problem
from tests.unit.solver.test_templates import _real_direct_allgather


pytestmark = pytest.mark.phase03


def _route_patterns(templates, channel_count=4):
    stats = RoutingModelStats(1, 1, 0, 0.0, 0.0)
    patterns = {}
    for template in templates:
        unit = template.representative
        root = unit.demands[0].root_rank
        parents = tuple(
            (root, rank)
            for rank in unit.node.communication_group
            if rank != root
        )
        selected = tuple(
            LinkKey(*unit.demands[0].physical_link(src, dst))
            for src, dst in parents
        )
        patterns[template.template_id] = RoutePattern(
            template_id=template.template_id,
            channel_count=channel_count,
            objective_mode=ObjectiveMode.LATENCY,
            selected_edges=selected,
            parent_edges=parents,
            model_stats=stats,
        )
    return patterns


def _direct_fixture():
    plan, problems = _real_direct_allgather(3, 2)
    templates = build_solver_templates(problems, plan.planning_mode)
    return problems, templates, _route_patterns(templates)


def test_instantiation_rebuilds_every_real_slice_with_provisional_allocations():
    problems, templates, patterns = _direct_fixture()

    result = instantiate_route_patterns(templates, patterns, problems)

    assert result.failures == ()
    assert set(result.node_schedules) == {
        problem.node.node_id for problem in problems
    }
    transfer_ids = []
    for problem in problems:
        schedule = result.node_schedules[problem.node.node_id]
        expected_members = frozenset(
            slice_id
            for demand in problem.demands
            for slice_id in demand.member_slice_ids
        )
        actual_members = frozenset(
            slice_id
            for transfer in schedule.transfers
            for slice_id in transfer.member_slice_ids
        )
        assert actual_members == expected_members
        assert schedule.metadata["routing_only"] is True
        assert schedule.metadata["channel_count"] == 4
        assert schedule.metadata["resource_slots"] == {
            transfer.transfer_id: {} for transfer in schedule.transfers
        }
        assert all(transfer.channel == 0 for transfer in schedule.transfers)
        assert all(
            transfer.st_time == transfer.ed_time == 0.0
            for transfer in schedule.transfers
        )
        for transfer in schedule.transfers:
            transfer_ids.append(transfer.transfer_id)
            for atom in transfer.atoms:
                assert atom.slice_id in expected_members
                assert atom.path[0].symbols[0].src_rank == source_rank(
                    atom.slice_id,
                    schedule.slice_count,
                )
                assert atom.current_symbol.dst_rank == transfer.dst_rank
                assert all(
                    symbol.ready_time == 0.0
                    for symbol in atom.path[0].symbols
                )
    assert len(transfer_ids) == len(set(transfer_ids))


def test_slice_specific_forbidden_mapping_falls_back_only_that_unit():
    problems, templates, patterns = _direct_fixture()
    template = next(item for item in templates if len(item.members) > 1)
    failed_member = template.members[1]
    problem_index = next(
        index
        for index, problem in enumerate(problems)
        if problem.node.node_id == failed_member.node_id
    )
    problem = problems[problem_index]
    unit = next(
        item
        for item in split_routing_units(problem)
        if item.unit_id == failed_member.unit_id
    )
    demand = unit.demands[0]
    parent_by_rank = {
        dict(failed_member.rank_map)[destination]: dict(failed_member.rank_map)[source]
        for source, destination in patterns[template.template_id].parent_edges
    }
    virtual = LinkKey(parent_by_rank[demand.required_leaf_rank], demand.required_leaf_rank)
    physical = LinkKey(*demand.physical_link(virtual.src_rank, virtual.dst_rank))
    forbidden = ForbiddenTransfer(
        slice_id=next(iter(demand.member_slice_ids)),
        src_rank=physical.src_rank,
        dst_rank=physical.dst_rank,
        stage_id=demand.stage_id,
    )
    changed_demand = replace(
        demand,
        forbidden_members=demand.forbidden_members + (forbidden,),
    )
    changed_problem = replace(
        problem,
        demands=tuple(
            changed_demand if item.demand_id == demand.demand_id else item
            for item in problem.demands
        ),
    )
    changed_problems = tuple(
        changed_problem if index == problem_index else item
        for index, item in enumerate(problems)
    )

    result = instantiate_route_patterns(templates, patterns, changed_problems)

    assert tuple(failure.unit_id for failure in result.failures) == (
        failed_member.unit_id,
    )
    assert result.failures[0].node_id == failed_member.node_id
    assert "forbidden" in result.failures[0].reason
    assert result.node_schedules[failed_member.node_id].transfers == ()
    assert all(
        result.node_schedules[member.node_id].transfers
        for current in templates
        for member in current.members
        if member.unit_id != failed_member.unit_id
    )


def test_stale_member_mapping_falls_back_only_that_unit():
    problems, templates, patterns = _direct_fixture()
    template = next(item for item in templates if len(item.members) > 1)
    failed_member = template.members[1]
    rank_map = list(failed_member.rank_map)
    rank_map[0] = (rank_map[0][0], 99)
    changed_member = replace(failed_member, rank_map=tuple(rank_map))
    changed_template = replace(
        template,
        members=tuple(
            changed_member if member == failed_member else member
            for member in template.members
        ),
    )
    changed_templates = tuple(
        changed_template if item == template else item for item in templates
    )

    result = instantiate_route_patterns(
        changed_templates,
        patterns,
        problems,
    )

    assert tuple(failure.unit_id for failure in result.failures) == (
        failed_member.unit_id,
    )
    assert "mapping target coverage" in result.failures[0].reason
    assert all(
        result.node_schedules[member.node_id].transfers
        for current in templates
        for member in current.members
        if member.unit_id != failed_member.unit_id
    )


def test_multihop_pattern_rebuilds_complete_member_path_prefixes():
    problems, templates, patterns = _direct_fixture()
    template = templates[0]
    representative = template.representative
    root = representative.demands[0].root_rank
    intermediate, leaf = sorted(
        set(representative.node.communication_group) - {root}
    )
    parent_edges = ((root, intermediate), (intermediate, leaf))
    patterns = dict(patterns)
    patterns[template.template_id] = replace(
        patterns[template.template_id],
        parent_edges=parent_edges,
        selected_edges=tuple(
            LinkKey(
                *representative.demands[0].physical_link(src_rank, dst_rank)
            )
            for src_rank, dst_rank in parent_edges
        ),
    )

    result = instantiate_route_patterns(templates, patterns, problems)

    assert result.failures == ()
    schedule = result.node_schedules[representative.node.node_id]
    final_transfer = next(
        transfer
        for transfer in schedule.transfers
        if schedule.metadata["route_unit_ids"][transfer.transfer_id]
        == representative.unit_id
        and transfer.dst_rank == leaf
    )
    assert all(
        tuple(
            (symbol.src_rank, symbol.dst_rank)
            for symbol in atom.path[0].symbols
        )
        == parent_edges
        for atom in final_transfer.atoms
    )


def test_instantiation_is_independent_of_template_and_problem_order():
    problems, templates, patterns = _direct_fixture()

    forward = instantiate_route_patterns(templates, patterns, problems)
    reversed_input = instantiate_route_patterns(
        tuple(reversed(templates)),
        patterns,
        tuple(reversed(problems)),
    )

    assert reversed_input == forward


def test_instantiation_result_keeps_its_schedule_mapping_immutable():
    problems, templates, patterns = _direct_fixture()
    result = instantiate_route_patterns(templates, patterns, problems)

    with pytest.raises(TypeError):
        result.node_schedules["replacement"] = next(
            iter(result.node_schedules.values())
        )


def test_reduction_dual_instantiation_rebuilds_real_reduce_semantics():
    problem = reduction_dual_problem()
    templates = build_solver_templates((problem,), PlanningMode.DIRECT)
    patterns = _route_patterns(templates, channel_count=2)

    result = instantiate_route_patterns(templates, patterns, (problem,))
    virtual = result.node_schedules[problem.node.node_id]
    reduced = reverse_allgather_schedule(
        virtual,
        problem.node.local_collective,
        problem.node.logical_output,
    )

    assert result.failures == ()
    assert virtual.metadata["routing_only"] is True
    assert virtual.metadata["reduction_dual"] is True
    assert {transfer.kind for transfer in virtual.transfers} == {"SEND"}
    assert {transfer.kind for transfer in reduced.transfers} == {"REDUCE"}
    assert reduced.transfers[0].member_slice_ids == frozenset({8})
    assert frozenset(
        reduced.metadata["tree_contributors"][reduced.transfers[0].transfer_id]
    ) == frozenset({0, 8})
