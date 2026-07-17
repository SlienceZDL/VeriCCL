import pytest

from vericcl.input.models import ObjectiveMode
from vericcl.solver.budget import ModelBudget
from vericcl.solver.milp import _build_model, _trees, solve_milp
from vericcl.topology.model import LinkKey

from tests.gurobi.helpers import (
    batching_problem,
    multihop_problem,
    require_gurobi_license,
    zero_duration_cycle_problem,
)


pytestmark = [pytest.mark.phase03, pytest.mark.gurobi]


@pytest.mark.parametrize("shared_resource", [False, True])
def test_multihop_transfer_preserves_ready_chain_and_resource_slots(
    shared_resource,
):
    require_gurobi_license()
    problem = multihop_problem(shared_resource=shared_resource)

    candidate = solve_milp(
        problem,
        channel_count=2,
        objective=ObjectiveMode.LATENCY,
        budget=ModelBudget(seconds=30, started_at=0, deadline=30),
        warm_start=None,
    )
    schedule = candidate.node_schedules[problem.node.node_id]
    first = next(
        transfer
        for transfer in schedule.transfers
        if (transfer.src_rank, transfer.dst_rank) == (0, 1)
    )
    second = next(
        transfer
        for transfer in schedule.transfers
        if (transfer.src_rank, transfer.dst_rank) == (1, 2)
    )

    assert first.ed_time <= second.st_time
    assert first.transfer_id in second.predecessor_ids
    assert second.atoms[0].path[0].symbols[1].ready_time == first.ed_time
    if shared_resource:
        slots = schedule.metadata["resource_slots"]
        assert slots[first.transfer_id]["shared-path"] == 0
        assert slots[second.transfer_id]["shared-path"] == 0


def test_batching_constrains_payloads_to_the_same_tree():
    require_gurobi_license()
    problem = batching_problem()

    candidate = solve_milp(
        problem,
        channel_count=2,
        objective=ObjectiveMode.LATENCY,
        budget=ModelBudget(seconds=30, started_at=0, deadline=30),
        warm_start=None,
    )
    schedule = candidate.node_schedules[problem.node.node_id]
    flows = schedule.metadata["selected_flows"]
    first_tree = set()
    second_tree = set()
    for demand_id, edges in flows.items():
        target = first_tree if "-a00000000-" in demand_id else second_tree
        target.update(edges)

    assert first_tree == second_tree
    assert candidate.search_space_restricted
    assert candidate.restrictions == ("batching",)


def test_tree_constraints_reject_a_disconnected_zero_duration_cycle():
    require_gurobi_license()
    problem = zero_duration_cycle_problem()
    gp = __import__("gurobipy")
    model, variables, _, _, _ = _build_model(
        gp,
        problem,
        _trees(problem),
        channel_count=1,
        objective=ObjectiveMode.LATENCY,
        budget=ModelBudget(seconds=30, started_at=0, deadline=30),
        warm_start=None,
    )
    demand_id = problem.demands[0].demand_id
    model.addConstr(
        variables.flow_selected[(demand_id, LinkKey(2, 3))] == 1
    )
    model.addConstr(
        variables.flow_selected[(demand_id, LinkKey(3, 2))] == 1
    )

    model.optimize()

    assert model.Status == gp.GRB.INFEASIBLE
    model.dispose()
