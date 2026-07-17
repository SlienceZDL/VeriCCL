from dataclasses import replace

import pytest

from vericcl.errors import SemanticError
from vericcl.input.models import ObjectiveMode
from vericcl.solver.model import (
    SolveCandidate,
    SolveStatus,
    SolverMetrics,
)
from vericcl.solver.objectives import (
    ObjectiveExpressions,
    configure_lexicographic_objective,
    rank_candidates,
)


pytestmark = pytest.mark.phase03


def _candidate(
    candidate_id,
    objective,
    makespan,
    operations,
    hops,
    load=0.0,
):
    metrics = SolverMetrics(
        status=SolveStatus.FEASIBLE,
        objective_values=(),
        best_bound=0.0,
        mip_gap=0.0,
        within_requested_gap=True,
        solve_time_s=1.0,
        model_count=1,
        operation_count=operations,
        hop_count=hops,
        makespan_us=makespan,
        maximum_normalized_resource_load=load,
        solver_name="test",
        solver_version="1",
        solver_seed=0,
        thread_count=1,
        termination_reason="complete",
    )
    return SolveCandidate(
        candidate_id=candidate_id,
        node_schedules={},
        objective_mode=objective,
        channel_count=1,
        metrics=metrics,
        selected_best=False,
        proven_optimal=False,
        search_space_restricted=False,
        restrictions=(),
        parent_candidate_id=None,
    )


def test_latency_tie_breaks_by_operations_then_hops():
    candidates = (
        _candidate("a", ObjectiveMode.LATENCY, 10, 8, 9),
        _candidate("b", ObjectiveMode.LATENCY, 10, 7, 11),
        _candidate("c", ObjectiveMode.LATENCY, 10, 7, 8),
    )

    ranked = rank_candidates(candidates)

    assert [item.candidate_id for item in ranked] == ["c", "b", "a"]


def test_throughput_ranks_load_before_makespan():
    candidates = (
        _candidate("fast", ObjectiveMode.THROUGHPUT, 8, 5, 5, load=0.8),
        _candidate("balanced", ObjectiveMode.THROUGHPUT, 10, 5, 5, load=0.6),
    )

    assert rank_candidates(candidates)[0].candidate_id == "balanced"


def test_stable_candidate_id_is_the_final_tie_break():
    first = _candidate("b", ObjectiveMode.LATENCY, 10, 5, 5)
    second = replace(first, candidate_id="a")

    assert rank_candidates((first, second))[0].candidate_id == "a"


def test_ranking_rejects_mixed_or_auto_objective_sets():
    latency = _candidate("latency", ObjectiveMode.LATENCY, 10, 5, 5)
    throughput = replace(
        latency,
        candidate_id="throughput",
        objective_mode=ObjectiveMode.THROUGHPUT,
    )
    automatic = replace(
        latency,
        candidate_id="auto",
        objective_mode=ObjectiveMode.AUTO,
    )

    with pytest.raises(SemanticError, match="objective"):
        rank_candidates((latency, throughput))
    with pytest.raises(SemanticError, match="AUTO"):
        rank_candidates((automatic,))


def test_empty_ranking_is_stable_and_invalid_members_are_rejected():
    assert rank_candidates(()) == ()
    with pytest.raises(SemanticError, match="SolveCandidate"):
        rank_candidates((object(),))


def test_objective_configuration_rejects_unresolved_modes():
    expressions = ObjectiveExpressions(object(), object(), object())

    with pytest.raises(SemanticError, match="ObjectiveMode"):
        configure_lexicographic_objective(None, None, "latency", expressions)
    with pytest.raises(SemanticError, match="AUTO"):
        configure_lexicographic_objective(
            None,
            None,
            ObjectiveMode.AUTO,
            expressions,
        )
    with pytest.raises(SemanticError, match="resource load"):
        configure_lexicographic_objective(
            None,
            None,
            ObjectiveMode.THROUGHPUT,
            expressions,
        )
