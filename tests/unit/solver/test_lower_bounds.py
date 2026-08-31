from dataclasses import replace

import pytest

from vericcl.errors import SemanticError, SolverUnavailableError
from vericcl.solver.lower_bounds import (
    LowerBound,
    dependency_time_lower_bound,
    global_throughput_time_lower_bound,
    throughput_time_lower_bound,
)

from tests.gurobi.helpers import broadcast_problem, multihop_problem


pytestmark = pytest.mark.phase03


def test_throughput_lower_bound_is_max_of_resource_and_dependency():
    bound = LowerBound(resource_us=80.0, dependency_us=95.0)

    assert bound.total_us == 95.0


def test_dependency_bound_keeps_multihop_causality_without_contention():
    problem = multihop_problem(shared_resource=False)

    bound = dependency_time_lower_bound(problem, max_channels=2)

    assert bound == 4.0


def test_resource_relaxation_and_dependency_bound_are_reported_separately(
    monkeypatch,
):
    problem = broadcast_problem(logical_positions=(0, 1))
    monkeypatch.setattr(
        "vericcl.solver.lower_bounds._resource_time_lower_bound",
        lambda current, channels: 4.0,
    )

    bound = throughput_time_lower_bound(problem, max_channels=1)

    assert bound.resource_us == 4.0
    assert bound.dependency_us == 2.0
    assert bound.total_us == max(bound.resource_us, bound.dependency_us)


def test_unavailable_lp_backend_falls_back_to_a_safe_zero_resource_bound(
    monkeypatch,
):
    problem = broadcast_problem(logical_positions=(0,))

    def unavailable():
        raise SolverUnavailableError("missing")

    monkeypatch.setattr(
        "vericcl.solver.lower_bounds.GurobiAdapter.require",
        unavailable,
    )

    bound = throughput_time_lower_bound(problem, max_channels=1)

    assert bound.resource_us == 0.0
    assert bound.dependency_us == 2.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("resource_us", -1),
        ("dependency_us", float("inf")),
        ("dependency_us", True),
    ],
)
def test_lower_bound_rejects_invalid_values(field, value):
    with pytest.raises(SemanticError):
        replace(LowerBound(1.0, 1.0), **{field: value})


def test_lower_bound_api_rejects_invalid_problem_and_channel_count():
    problem = broadcast_problem(logical_positions=(0,))

    with pytest.raises(SemanticError, match="problem"):
        dependency_time_lower_bound(object(), max_channels=1)
    with pytest.raises(SemanticError, match="positive integer"):
        dependency_time_lower_bound(problem, max_channels=0)
    with pytest.raises(SemanticError, match="problem"):
        throughput_time_lower_bound(object(), max_channels=1)


def test_global_lower_bound_keeps_dependency_causality_without_lp(monkeypatch):
    problem = multihop_problem(shared_resource=True)

    def unavailable():
        raise SolverUnavailableError("missing")

    monkeypatch.setattr(
        "vericcl.solver.lower_bounds.GurobiAdapter.require",
        unavailable,
    )

    bound = global_throughput_time_lower_bound(
        (problem, problem),
        max_channels=2,
    )

    assert bound.resource_us == 0.0
    assert bound.dependency_us == 4.0
