import pytest

from vericcl.input.models import ObjectiveMode
from vericcl.solver.budget import ModelBudget
from vericcl.solver.routing_milp import solve_route_milp
from vericcl.solver.templates import build_solver_templates

from tests.gurobi.helpers import require_gurobi_license
from tests.unit.solver.test_templates import _real_direct_allgather


pytestmark = [pytest.mark.phase03, pytest.mark.gurobi]


def test_route_model_size_is_exactly_invariant_across_real_slice_counts():
    require_gurobi_license()
    counts = []
    member_counts = []
    for slice_count in (8, 16, 64, 128):
        plan, problems = _real_direct_allgather(8, slice_count)
        template = next(
            item
            for item in build_solver_templates(problems, plan.planning_mode)
            if item.representative.demands[0].root_rank == 0
        )
        pattern = solve_route_milp(
            template,
            channel_count=4,
            objective_mode=ObjectiveMode.LATENCY,
            budget=ModelBudget(seconds=30, started_at=0, deadline=30),
        )
        member_counts.append(len(template.members))
        counts.append(
            (
                pattern.model_stats.variable_count,
                pattern.model_stats.constraint_count,
                pattern.model_stats.general_constraint_count,
            )
        )

    assert member_counts == [8, 16, 64, 128]
    assert counts[1:] == [counts[0], counts[0], counts[0]]
