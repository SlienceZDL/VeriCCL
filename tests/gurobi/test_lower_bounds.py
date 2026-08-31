from dataclasses import replace

import pytest

from vericcl.planner.model import PlanningMode
from vericcl.solver.demands import build_solver_problem
from vericcl.solver.lower_bounds import (
    global_throughput_time_lower_bound,
    throughput_time_lower_bound,
)
from vericcl.solver.templates import build_solver_templates

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


def test_global_resource_bound_accumulates_shared_load_across_plan_nodes():
    require_gurobi_license()
    problem = multihop_problem(shared_resource=True)

    bound = global_throughput_time_lower_bound(
        (problem, problem),
        max_channels=2,
    )

    assert bound.resource_us == 8.0
    assert bound.dependency_us == 4.0


def test_template_multiplicity_counts_every_real_payload_tree():
    require_gurobi_license()
    original = broadcast_problem(logical_positions=tuple(range(128)))
    inputs = replace(
        original.inputs,
        hyperparameters=replace(
            original.inputs.hyperparameters,
            total_size_bytes=(
                128 * original.inputs.hyperparameters.slice_size_bytes
            ),
        ),
    )
    problem = build_solver_problem(
        original.node,
        inputs,
        original.topology,
    )
    templates = build_solver_templates(
        (problem,),
        PlanningMode.DIRECT,
    )

    bound = global_throughput_time_lower_bound(
        (problem,),
        max_channels=1,
    )

    assert len(templates) == 1
    assert len(templates[0].members) == 128
    assert bound.resource_us == 256.0
    assert bound.dependency_us == 2.0
