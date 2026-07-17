import pytest

from vericcl.errors import SemanticError
from vericcl.input.models import ObjectiveMode
from vericcl.solver.budget import ModelBudget
from vericcl.solver.milp import solve_milp

from tests.gurobi.helpers import broadcast_problem


pytestmark = pytest.mark.phase03


def _arguments():
    return {
        "problem": broadcast_problem(logical_positions=(0,)),
        "channel_count": 1,
        "objective": ObjectiveMode.LATENCY,
        "budget": ModelBudget(seconds=1, started_at=0, deadline=1),
        "warm_start": None,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("problem", object()),
        ("channel_count", 0),
        ("channel_count", 33),
        ("objective", "latency"),
        ("budget", object()),
        ("warm_start", object()),
    ],
)
def test_solve_milp_rejects_invalid_api_arguments(field, value):
    arguments = _arguments()
    arguments[field] = value

    with pytest.raises(SemanticError):
        solve_milp(**arguments)
