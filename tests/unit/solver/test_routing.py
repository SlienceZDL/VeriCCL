from dataclasses import FrozenInstanceError, replace

import pytest

from vericcl.errors import SemanticError
from vericcl.input.json_codec import canonical_json
from vericcl.input.models import ObjectiveMode
from vericcl.solver.model import SolveStatus, SolverMetrics
from vericcl.solver.routing import RoutePattern, RoutingModelStats


pytestmark = pytest.mark.phase03


def _metrics():
    return SolverMetrics(
        status=SolveStatus.OPTIMAL,
        objective_values=(4.0, 2.0, 3.0),
        best_bound=4.0,
        mip_gap=0.0,
        within_requested_gap=True,
        solve_time_s=0.25,
        model_count=1,
        operation_count=2,
        hop_count=3,
        makespan_us=4.0,
        maximum_normalized_resource_load=2.0,
        solver_name="gurobi",
        solver_version="test",
        solver_seed=0,
        thread_count=1,
        termination_reason="optimal",
    )


def _stats():
    return RoutingModelStats(
        variable_count=12,
        constraint_count=20,
        general_constraint_count=0,
        build_time_s=0.1,
        optimize_time_s=0.15,
    )


def _pattern():
    return RoutePattern(
        template_id="template-a",
        channel_count=4,
        objective_mode=ObjectiveMode.LATENCY,
        selected_edges=((1, 2), (0, 1)),
        member_paths=(
            ("leaf-b", ((0, 1), (1, 2))),
            ("leaf-a", ((0, 1),)),
        ),
        metrics=_metrics(),
        model_stats=_stats(),
    )


def test_route_pattern_is_immutable_and_canonicalizes_unordered_sets():
    pattern = _pattern()

    assert pattern.selected_edges == ((0, 1), (1, 2))
    assert tuple(key for key, _ in pattern.member_paths) == (
        "leaf-a",
        "leaf-b",
    )
    with pytest.raises(FrozenInstanceError):
        pattern.channel_count = 2
    with pytest.raises(FrozenInstanceError):
        pattern.model_stats.variable_count = 1


def test_route_pattern_has_stable_canonical_json():
    first = _pattern()
    second = replace(
        first,
        selected_edges=tuple(reversed(first.selected_edges)),
        member_paths=tuple(reversed(first.member_paths)),
    )

    assert canonical_json(first) == canonical_json(second)


def test_route_pattern_rejects_discontinuous_or_cyclic_member_paths():
    pattern = _pattern()

    with pytest.raises(SemanticError, match="continuous"):
        replace(
            pattern,
            selected_edges=((0, 1), (2, 3)),
            member_paths=(("leaf-a", ((0, 1), (2, 3))),),
        )
    with pytest.raises(SemanticError, match="cycle"):
        replace(
            pattern,
            selected_edges=((0, 1), (1, 0)),
            member_paths=(("leaf-a", ((0, 1), (1, 0))),),
        )


def test_route_pattern_rejects_paths_outside_the_selected_tree():
    pattern = _pattern()

    with pytest.raises(SemanticError, match="selected tree"):
        replace(
            pattern,
            selected_edges=((0, 1),),
            member_paths=(("leaf-a", ((0, 1), (1, 2))),),
        )


def test_route_pattern_rejects_auto_and_duplicate_members():
    pattern = _pattern()

    with pytest.raises(SemanticError, match="AUTO"):
        replace(pattern, objective_mode=ObjectiveMode.AUTO)
    with pytest.raises(SemanticError, match="unique"):
        replace(
            pattern,
            member_paths=(
                ("leaf-a", ((0, 1),)),
                ("leaf-a", ((0, 1),)),
            ),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("variable_count", -1),
        ("constraint_count", True),
        ("general_constraint_count", -1),
        ("build_time_s", float("inf")),
        ("optimize_time_s", -0.1),
    ],
)
def test_routing_model_stats_reject_invalid_measurements(field, value):
    with pytest.raises(SemanticError, match=field):
        replace(_stats(), **{field: value})
