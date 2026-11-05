import pytest

from vericcl.errors import SemanticError
from vericcl.solver.budget import ModelBudget, SolveBudget


pytestmark = pytest.mark.phase03


def test_model_budget_is_bounded_by_both_deadlines():
    budget = SolveBudget(
        total_seconds=100,
        per_model_seconds=30,
        started_at=0,
    )

    model = budget.model_budget(now=85)

    assert model.seconds == 15
    assert model.deadline == 100


def test_model_budget_uses_per_model_limit_when_total_time_is_available():
    budget = SolveBudget(
        total_seconds=100,
        per_model_seconds=30,
        started_at=10,
    )

    model = budget.model_budget(now=20)

    assert model.seconds == 30
    assert model.deadline == 50


def test_expired_total_budget_does_not_start_positive_model_budget():
    budget = SolveBudget(
        total_seconds=10,
        per_model_seconds=30,
        started_at=0,
    )

    assert budget.expired(now=10)
    assert budget.remaining_seconds(now=11) == 0
    assert budget.model_budget(now=11).seconds == 0


def test_budget_uses_injected_monotonic_clock():
    values = iter((10.0, 12.5, 15.0))
    budget = SolveBudget(
        total_seconds=10,
        per_model_seconds=3,
        clock=lambda: next(values),
    )

    assert budget.started_at == 10.0
    assert budget.remaining_seconds() == 7.5
    assert budget.model_budget().started_at == 15.0


@pytest.mark.parametrize(
    "total,per_model",
    [(0, 1), (1, 0), (float("inf"), 1), (1, True)],
)
def test_solve_budget_rejects_invalid_limits(total, per_model):
    with pytest.raises(SemanticError):
        SolveBudget(total_seconds=total, per_model_seconds=per_model)


def test_model_budget_rejects_inconsistent_deadline():
    with pytest.raises(SemanticError, match="deadline"):
        ModelBudget(seconds=5, started_at=10, deadline=14)


@pytest.mark.parametrize(
    "seconds,started,deadline",
    [(-1, 0, -1), (1, True, 1), (1, 0, float("nan"))],
)
def test_model_budget_rejects_invalid_values(seconds, started, deadline):
    with pytest.raises(SemanticError):
        ModelBudget(seconds=seconds, started_at=started, deadline=deadline)


def test_solve_budget_rejects_non_callable_clock():
    with pytest.raises(SemanticError, match="clock"):
        SolveBudget(total_seconds=10, per_model_seconds=2, clock=None)


def test_solve_budget_rejects_non_finite_current_time():
    budget = SolveBudget(total_seconds=10, per_model_seconds=2, started_at=0)

    with pytest.raises(SemanticError):
        budget.remaining_seconds(now=float("nan"))
