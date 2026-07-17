import pytest

from vericcl.solver.lower_bounds import throughput_time_lower_bound

from tests.gurobi.helpers import (
    broadcast_problem,
    multihop_problem,
    require_gurobi_license,
)


pytestmark = [pytest.mark.phase03, pytest.mark.gurobi]


def test_continuous_resource_relaxation_counts_each_payload_tree_once():
    require_gurobi_license()
    problem = broadcast_problem(logical_positions=(0, 1))

    bound = throughput_time_lower_bound(problem, max_channels=1)

    assert bound.resource_us == 4.0
    assert bound.dependency_us == 2.0


def test_shared_resource_capacity_is_included_in_the_relaxation():
    require_gurobi_license()
    problem = multihop_problem(shared_resource=True)

    bound = throughput_time_lower_bound(problem, max_channels=2)

    assert bound.resource_us == 4.0
    assert bound.dependency_us == 4.0
