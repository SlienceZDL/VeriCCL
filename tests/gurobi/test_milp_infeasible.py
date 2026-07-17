import pytest

from vericcl.input.models import ForbiddenTransfer, ObjectiveMode
from vericcl.solver.budget import ModelBudget
from vericcl.solver.milp import solve_milp
from vericcl.solver.model import SolveStatus

from tests.gurobi.helpers import broadcast_problem, require_gurobi_license


pytestmark = [pytest.mark.phase03, pytest.mark.gurobi]


def test_forbidden_only_link_returns_infeasible_without_schedule():
    require_gurobi_license()
    problem = broadcast_problem(
        logical_positions=(0,),
        forbidden=(ForbiddenTransfer(0, 0, 1, 0),),
    )

    candidate = solve_milp(
        problem,
        channel_count=1,
        objective=ObjectiveMode.LATENCY,
        budget=ModelBudget(seconds=30, started_at=0, deadline=30),
        warm_start=None,
    )

    assert candidate.metrics.status is SolveStatus.INFEASIBLE
    assert not candidate.node_schedules
    assert not candidate.proven_optimal


def test_expired_model_budget_does_not_build_a_schedule():
    require_gurobi_license()
    problem = broadcast_problem(logical_positions=(0,))

    candidate = solve_milp(
        problem,
        channel_count=1,
        objective=ObjectiveMode.LATENCY,
        budget=ModelBudget(seconds=0, started_at=5, deadline=5),
        warm_start=None,
    )

    assert candidate.metrics.status is SolveStatus.NOT_RUN
    assert candidate.metrics.termination_reason == "budget_exhausted"
    assert not candidate.node_schedules
