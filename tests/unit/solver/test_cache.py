from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

import pytest

from vericcl.errors import SemanticError
from vericcl.input.loader import resolve_inputs
from vericcl.input.json_codec import canonical_json
from vericcl.planner.build import build_plan
from vericcl.planner.model import PlanningMode
from vericcl.solver.cache import (
    CacheSignature,
    CandidateCache,
    _CacheEntry,
    build_cache_signature,
    candidate_cache_key,
    performance_cache_key,
    structural_cache_key,
)
from vericcl.solver.demands import build_solver_problem
from vericcl.solver.model import SearchDiagnostics, SolveRequest, SolveStatus
from vericcl.solver.orchestrator import solve
from vericcl.solver.templates import build_solver_templates
from vericcl.topology.loader import load_topology
from vericcl.topology.model import DirectedLink
from vericcl.tuning.model import TuningOverlay

from tests.unit.solver.test_model import candidate


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


def test_cache_key_is_restart_stable_across_hash_seeds():
    script = """
from tests.unit.solver.test_cache import request
from vericcl.solver.cache import build_cache_signature, candidate_cache_key
from vericcl.solver.demands import build_solver_problem
from vericcl.solver.templates import build_solver_templates

value = request()
problems = tuple(
    build_solver_problem(node, value.inputs, value.topology)
    for node in value.plan.nodes
)
templates = build_solver_templates(problems, value.plan.planning_mode)
print(candidate_cache_key(
    value,
    build_cache_signature(value, problems, templates),
))
"""
    keys = []
    for hash_seed in ("1", "987654"):
        environment = dict(os.environ, PYTHONHASHSEED=hash_seed)
        keys.append(
            subprocess.check_output(
                (sys.executable, "-c", script),
                cwd=Path(__file__).parents[3],
                env=environment,
                text=True,
            ).strip()
        )

    assert keys[0] == keys[1]


def test_cache_key_versions_are_explicit_and_independent():
    value = request()
    problems = tuple(
        build_solver_problem(node, value.inputs, value.topology)
        for node in value.plan.nodes
    )
    templates = build_solver_templates(problems, value.plan.planning_mode)
    signature = build_cache_signature(value, problems, templates)

    assert candidate_cache_key(value, signature) != candidate_cache_key(
        value,
        replace(signature, route_model_version="2"),
    )
    assert candidate_cache_key(value, signature) != candidate_cache_key(
        value,
        replace(signature, global_scheduler_version="2"),
    )
    assert signature.planning_mode == value.plan.planning_mode.value
    assert candidate_cache_key(value, signature) != candidate_cache_key(
        value,
        replace(signature, planning_mode="manual"),
    )
    assert CacheSignature().backend_type == "template_route"
    assert signature.problem_count == len(problems)
    assert signature.template_count == len(templates)
    assert signature.template_member_count == sum(
        len(template.members) for template in templates
    )
    assert len(signature.structure_digest_sha256) == 64
    assert len(signature.problem_digest_sha256) == 64
    assert len(signature.template_digest_sha256) == 64


def test_cache_signature_serialized_size_is_bounded_across_slice_counts():
    sizes = []
    for slice_count in (2, 8, 32, 128, 512):
        value = request()
        inputs = replace(
            value.inputs,
            hyperparameters=replace(
                value.inputs.hyperparameters,
                total_size_bytes=(
                    slice_count
                    * value.inputs.hyperparameters.slice_size_bytes
                ),
            ),
        )
        value = replace(
            value,
            inputs=inputs,
            plan=build_plan(inputs, value.topology),
        )
        problems = tuple(
            build_solver_problem(node, value.inputs, value.topology)
            for node in value.plan.nodes
        )
        templates = build_solver_templates(
            problems,
            value.plan.planning_mode,
        )

        signature = build_cache_signature(value, problems, templates)
        sizes.append(len(canonical_json(signature)))

    assert max(sizes) < 700
    assert max(sizes) - min(sizes) < 32


def test_cache_key_tracks_exact_templates_and_member_mappings_stably():
    value = request()
    problems = tuple(
        build_solver_problem(node, value.inputs, value.topology)
        for node in value.plan.nodes
    )
    templates = build_solver_templates(problems, value.plan.planning_mode)
    original = build_cache_signature(value, problems, templates)
    changed_exact = (
        replace(templates[0], exact_signature="f" * 64),
        *templates[1:],
    )
    member = templates[0].members[0]
    changed_member = replace(
        member,
        rank_map=tuple(
            (source, member.rank_map[-index - 1][1])
            for index, (source, _) in enumerate(member.rank_map)
        ),
    )
    changed_mapping = (
        replace(
            templates[0],
            members=(changed_member, *templates[0].members[1:]),
        ),
        *templates[1:],
    )

    assert candidate_cache_key(value, original) != candidate_cache_key(
        value,
        build_cache_signature(value, problems, changed_exact),
    )
    assert candidate_cache_key(value, original) != candidate_cache_key(
        value,
        build_cache_signature(value, problems, changed_mapping),
    )
    assert original == build_cache_signature(
        value,
        tuple(reversed(problems)),
        tuple(reversed(templates)),
    )


def test_solver_uses_actual_template_signature_for_cache_entries():
    value = request()
    inputs = replace(
        value.inputs,
        strategies=replace(value.inputs.strategies, milp=False),
    )
    value = replace(
        value,
        inputs=inputs,
        plan=build_plan(inputs, value.topology),
    )
    problems = tuple(
        build_solver_problem(node, value.inputs, value.topology)
        for node in value.plan.nodes
    )
    templates = build_solver_templates(problems, value.plan.planning_mode)
    signature = build_cache_signature(value, problems, templates)
    cache = CandidateCache()

    result = solve(value, cache=cache)

    assert result.selected_candidate is not None
    expected_key = candidate_cache_key(value, signature)
    assert cache.get(expected_key) is not None
    assert result.selected_candidate.candidate_id.endswith(expected_key[:12])


def test_legacy_and_template_cache_signatures_are_isolated():
    value = request()
    problems = tuple(
        build_solver_problem(node, value.inputs, value.topology)
        for node in value.plan.nodes
    )
    templates = build_solver_templates(problems, value.plan.planning_mode)
    template_signature = build_cache_signature(value, problems, templates)
    strict_inputs = replace(
        value.inputs,
        solver=replace(value.inputs.solver, require_proven_optimal=True),
    )
    strict = replace(value, inputs=strict_inputs)
    strict_problems = tuple(
        build_solver_problem(node, strict.inputs, strict.topology)
        for node in strict.plan.nodes
    )
    legacy_signature = build_cache_signature(strict, strict_problems, ())
    cache = CandidateCache()
    cache.put(
        candidate_cache_key(strict, legacy_signature),
        candidate(),
        ttl_seconds=10,
        complete=True,
        now=0,
    )

    assert template_signature.backend_type == "template_route"
    assert legacy_signature.backend_type == "legacy_full_time_milp"
    assert candidate_cache_key(value, template_signature) != candidate_cache_key(
        strict,
        legacy_signature,
    )
    assert (
        cache.get(candidate_cache_key(value, template_signature), now=1)
        is None
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"planning_mode": PlanningMode.MANUAL},
        {"planning_reason": "no_eligible_gateway_domain"},
    ],
)
def test_planning_metadata_changes_structural_cache_key(changes):
    original = request()
    updated = replace(original, plan=replace(original.plan, **changes))

    assert structural_cache_key(original) != structural_cache_key(updated)


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


def test_typed_cache_entry_round_trips_candidate_and_diagnostics():
    cache = CandidateCache()
    value = candidate()
    diagnostics = SearchDiagnostics(
        requested_problem_count=2,
        route_model_count=3,
        search_model_count_total=3,
        route_model_build_time_s=1.25,
        route_model_optimize_time_s=2.5,
        model_variables_max=17,
    )
    cache.put(
        "key",
        value,
        ttl_seconds=10,
        complete=True,
        now=0,
        diagnostics=diagnostics,
    )

    cached = cache.get_entry("key", now=1)

    assert cached is not None
    assert cached.candidate is value
    assert cached.diagnostics == diagnostics


def test_candidate_only_cache_entry_reads_with_default_diagnostics():
    cache = CandidateCache()
    value = candidate()
    cache._entries["key"] = _CacheEntry(
        value=value,
        expires_at=10.0,
        complete=True,
    )

    cached = cache.get_entry("key", now=1)

    assert cached is not None
    assert cached.candidate is value
    assert cached.diagnostics == SearchDiagnostics()


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
