from dataclasses import replace
from pathlib import Path

import pytest

from vericcl.errors import SemanticError
from vericcl.input.loader import resolve_inputs
from vericcl.input.models import ObjectiveMode
from vericcl.planner.build import build_plan
from vericcl.planner.model import PlanningMode
from vericcl.solver.cache import (
    CandidateCache,
    candidate_cache_key,
    performance_cache_key,
    route_model_cache_key,
    structural_cache_key,
)
from vericcl.solver.model import (
    SearchDiagnostics,
    SolveRequest,
    SolveResult,
    SolveStatus,
)
from vericcl.solver.templates import build_solver_templates
from vericcl.topology.loader import load_topology
from vericcl.topology.model import DirectedLink
from vericcl.tuning.model import TuningOverlay

from tests.unit.solver.test_model import candidate
from tests.unit.solver.test_templates import _real_direct_allgather


pytestmark = pytest.mark.phase03


EXAMPLES = Path(__file__).parents[3] / "vericcl" / "examples"


def request(seed=0, environment="env-a"):
    inputs = resolve_inputs(
        EXAMPLES / "topo" / "two_rank.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    inputs = replace(
        inputs,
        solver=replace(inputs.solver, solver_seed=seed),
    )
    topology = load_topology(inputs)
    return SolveRequest(
        inputs=inputs,
        topology=topology,
        plan=build_plan(inputs, topology),
        solver_version="solver-1",
        model_version="model-1",
        environment_signature=environment,
    )


def test_solver_seed_changes_cache_key():
    assert candidate_cache_key(request(seed=0)) != candidate_cache_key(
        request(seed=1)
    )


def test_environment_changes_only_performance_cache_key():
    left = request(environment="env-a")
    right = replace(left, environment_signature="env-b")

    assert structural_cache_key(left) == structural_cache_key(right)
    assert performance_cache_key(left) != performance_cache_key(right)


def test_performance_update_preserves_structural_key():
    original = request()
    key, edge = next(iter(original.topology.links.items()))
    updated_edge = DirectedLink(
        key=key,
        max_channels=edge.max_channels,
        performance=replace(edge.performance, invbw_us=3.0),
        resource_ids=edge.resource_ids,
    )
    topology = replace(
        original.topology,
        links={
            current_key: updated_edge if current_key == key else current_edge
            for current_key, current_edge in original.topology.links.items()
        },
        isomorphism_signature="",
    )
    updated = replace(original, topology=topology)

    assert structural_cache_key(original) == structural_cache_key(updated)
    assert performance_cache_key(original) != performance_cache_key(updated)


def test_overlay_channel_count_changes_candidate_key():
    base = request()
    tuned = replace(
        base,
        overlay=TuningOverlay(
            overlay_id="overlay",
            parent_candidate_id=None,
            channel_count=2,
        ),
    )

    assert candidate_cache_key(base) != candidate_cache_key(tuned)


def test_cache_key_is_stable_for_equivalent_request():
    value = request()

    assert candidate_cache_key(value) == candidate_cache_key(replace(value))


def test_planning_metadata_changes_cache_key():
    value = request()

    changed_mode = replace(
        value,
        plan=replace(
            value.plan,
            planning_mode=PlanningMode.MANUAL,
        ),
    )
    changed_reason = replace(
        value,
        plan=replace(
            value.plan,
            planning_reason="manual_hierarchy",
        ),
    )

    assert candidate_cache_key(value) != candidate_cache_key(changed_mode)
    assert candidate_cache_key(value) != candidate_cache_key(changed_reason)


def test_expired_cache_entry_is_not_returned():
    cache = CandidateCache()
    cache.put("key", candidate(), ttl_seconds=10, complete=True, now=0)

    assert cache.get("key", now=10) is None


def test_partial_cache_entry_does_not_retain_optimality_proof():
    cache = CandidateCache()
    optimal = candidate(proven_optimal=True, status=SolveStatus.OPTIMAL)
    cache.put("key", optimal, ttl_seconds=10, complete=False, now=0)

    cached = cache.get("key", now=1)

    assert cached is not None
    assert not cached.proven_optimal


def test_complete_cache_entry_is_returned_unchanged():
    cache = CandidateCache()
    value = candidate()
    cache.put("key", value, ttl_seconds=10, complete=True, now=0)

    assert cache.get("key", now=1) is value
    assert cache.get("missing", now=1) is None


def test_cache_entry_preserves_optional_search_diagnostics():
    cache = CandidateCache()
    diagnostics = SearchDiagnostics(
        requested_problem_count=2,
        template_count=1,
        route_model_count=4,
    )
    cache.put(
        "key",
        candidate(),
        ttl_seconds=10,
        complete=True,
        now=0,
        diagnostics=diagnostics,
    )

    cached, cached_diagnostics = cache.get_with_diagnostics("key", now=1)

    assert cached is not None
    assert cached_diagnostics == diagnostics


def test_old_solve_result_construction_defaults_search_diagnostics():
    result = SolveResult(
        status=SolveStatus.NOT_RUN,
        candidates=(),
        selected_candidate_id=None,
        cache_hit=False,
        message="old-payload",
    )

    assert result.diagnostics == SearchDiagnostics()


def test_route_model_cache_key_binds_exact_search_identity():
    plan, problems = _real_direct_allgather(3, 2)
    templates = build_solver_templates(problems, plan.planning_mode)
    template = next(item for item in templates if len(item.members) > 1)
    base = route_model_cache_key(
        plan.planning_mode,
        template,
        ObjectiveMode.LATENCY,
        1,
    )
    changed_member = replace(template.members[1], node_id="changed-member")
    changed_template = replace(
        template,
        members=tuple(
            changed_member if member == template.members[1] else member
            for member in template.members
        ),
    )

    assert base != route_model_cache_key(
        PlanningMode.MANUAL,
        template,
        ObjectiveMode.LATENCY,
        1,
    )
    assert base != route_model_cache_key(
        plan.planning_mode,
        changed_template,
        ObjectiveMode.LATENCY,
        1,
    )
    assert base != route_model_cache_key(
        plan.planning_mode,
        template,
        ObjectiveMode.THROUGHPUT,
        1,
    )
    assert base != route_model_cache_key(
        plan.planning_mode,
        template,
        ObjectiveMode.LATENCY,
        2,
    )
    assert base != route_model_cache_key(
        plan.planning_mode,
        template,
        ObjectiveMode.LATENCY,
        1,
        route_model_version="changed-route-model",
    )
    assert base != route_model_cache_key(
        plan.planning_mode,
        template,
        ObjectiveMode.LATENCY,
        1,
        global_scheduler_version="changed-global-scheduler",
    )


@pytest.mark.parametrize("key_function", [structural_cache_key, performance_cache_key])
def test_cache_key_functions_require_solve_request(key_function):
    with pytest.raises(SemanticError, match="SolveRequest"):
        key_function(object())


@pytest.mark.parametrize(
    "field,value",
    [
        ("key", ""),
        ("candidate", object()),
        ("ttl_seconds", 0),
        ("ttl_seconds", float("nan")),
        ("complete", 1),
        ("now", float("nan")),
    ],
)
def test_cache_rejects_invalid_entries(field, value):
    arguments = {
        "key": "key",
        "candidate": candidate(),
        "ttl_seconds": 10,
        "complete": True,
        "now": 0,
    }
    arguments[field] = value

    with pytest.raises(SemanticError):
        CandidateCache().put(**arguments)


def test_cache_rejects_non_finite_lookup_time():
    with pytest.raises(SemanticError):
        CandidateCache().get("key", now=float("inf"))
