import math
import time
from collections import defaultdict
from dataclasses import replace
from typing import Dict, Mapping, Optional, Tuple

from vericcl.errors import (
    ConstructionInfeasibleError,
    SemanticError,
    SolverUnavailableError,
    VeriCCLError,
)
from vericcl.input.models import ObjectiveMode, ResolvedInput
from vericcl.input.json_codec import sha256_json
from vericcl.planner.build import build_plan
from vericcl.semantics.atom import Schedule
from vericcl.solver.cache import (
    CandidateCache,
    candidate_cache_key,
    route_model_cache_key,
)
from vericcl.solver.constructive import construct_candidate
from vericcl.solver.demands import SolverProblem, build_solver_problem
from vericcl.solver.gurobi_api import GurobiAdapter
from vericcl.solver.lower_bounds import throughput_time_lower_bound
from vericcl.solver.model import (
    SearchDiagnostics,
    SolveCandidate,
    SolveRequest,
    SolveResult,
    SolveStatus,
    SolverMetrics,
)
from vericcl.solver.objectives import rank_candidates
from vericcl.solver.instantiate import instantiate_route_patterns
from vericcl.solver.routing import RoutePattern
from vericcl.solver.search import (
    RouteSearchResult,
    search_models,
    search_route_models,
)
from vericcl.solver.templates import (
    RoutingUnit,
    SolverTemplate,
    TemplateMember,
    build_solver_templates,
    split_routing_units,
)


_CACHE_TTL_SECONDS = 3600.0
_DEFAULT_CACHE = CandidateCache()
_monotonic = time.monotonic
_diagnostic_clock = time.perf_counter


def _makespan(schedule: Schedule) -> float:
    return max(
        (transfer.ed_time for transfer in schedule.transfers),
        default=0.0,
    )


def _maximum_resource_load(schedule: Schedule) -> float:
    loads = defaultdict(float)
    raw_slots = schedule.metadata.get("resource_slots", {})
    if not isinstance(raw_slots, Mapping):
        raise SemanticError("resource_slots metadata must be a mapping")
    for transfer in schedule.transfers:
        duration = transfer.ed_time - transfer.st_time
        loads[("link", transfer.src_rank, transfer.dst_rank)] += duration
        slots = raw_slots.get(transfer.transfer_id, {})
        if not isinstance(slots, Mapping):
            raise SemanticError("transfer resource slots must be a mapping")
        for resource_id in slots:
            loads[("resource", resource_id)] += duration
    return max(loads.values(), default=0.0)


def _objective_values(
    objective: ObjectiveMode,
    makespan_us: float,
    resource_load_us: float,
    operation_count: int,
    hop_count: int,
) -> Tuple[float, ...]:
    if objective is ObjectiveMode.LATENCY:
        return (
            makespan_us,
            float(operation_count),
            float(hop_count),
        )
    return (resource_load_us, makespan_us)


def _constructive_candidate(
    problem: SolverProblem,
    schedule: Schedule,
    objective: ObjectiveMode,
    channel_count: int,
    parent_candidate_id: Optional[str],
) -> SolveCandidate:
    makespan_us = _makespan(schedule)
    resource_load_us = _maximum_resource_load(schedule)
    operation_count = len(schedule.transfers)
    return SolveCandidate(
        candidate_id="{}-constructive-{}-k{:02d}".format(
            problem.node.node_id,
            objective.value,
            channel_count,
        ),
        node_schedules={problem.node.node_id: schedule},
        objective_mode=objective,
        channel_count=channel_count,
        metrics=SolverMetrics(
            status=SolveStatus.FEASIBLE,
            objective_values=_objective_values(
                objective,
                makespan_us,
                resource_load_us,
                operation_count,
                operation_count,
            ),
            best_bound=0.0,
            mip_gap=0.0,
            within_requested_gap=False,
            solve_time_s=0.0,
            model_count=0,
            operation_count=operation_count,
            hop_count=operation_count,
            makespan_us=makespan_us,
            maximum_normalized_resource_load=resource_load_us,
            solver_name="constructive",
            solver_version="1",
            solver_seed=problem.inputs.solver.solver_seed,
            thread_count=1,
            termination_reason="constructive_complete",
        ),
        selected_best=False,
        proven_optimal=False,
        search_space_restricted=problem.search_space_restricted,
        restrictions=problem.restrictions,
        parent_candidate_id=parent_candidate_id,
    )


def _combined_status(candidates: Tuple[SolveCandidate, ...]) -> SolveStatus:
    statuses = {candidate.metrics.status for candidate in candidates}
    if statuses == {SolveStatus.OPTIMAL}:
        return SolveStatus.OPTIMAL
    if SolveStatus.TIME_LIMIT in statuses:
        return SolveStatus.TIME_LIMIT
    return SolveStatus.FEASIBLE


def _common_text(values, fallback: str) -> str:
    unique = sorted(set(values))
    return unique[0] if len(unique) == 1 else fallback


def _combine_node_candidates(
    request: SolveRequest,
    candidates: Mapping[str, SolveCandidate],
    backend: str,
    objective: ObjectiveMode,
    channel_count: int,
) -> SolveCandidate:
    from vericcl.composer import compose

    expected = {node.node_id for node in request.plan.nodes}
    if set(candidates) != expected:
        raise SemanticError("one local candidate is required for every plan node")
    ordered = tuple(candidates[node.node_id] for node in request.plan.nodes)
    if any(candidate.channel_count != channel_count for candidate in ordered):
        raise SemanticError("combined candidates must use one channel count")
    global_schedule = compose(request.plan, candidates)
    makespan_us = _makespan(global_schedule)
    resource_load_us = _maximum_resource_load(global_schedule)
    operation_count = len(global_schedule.transfers)
    restrictions = {
        restriction
        for candidate in ordered
        for restriction in candidate.restrictions
    }
    if len(request.plan.nodes) > 1:
        restrictions.add("independent_node_composition")
    status = _combined_status(ordered)
    primary_value = (
        makespan_us
        if objective is ObjectiveMode.LATENCY
        else resource_load_us
    )
    best_bound = max(
        (candidate.metrics.best_bound for candidate in ordered),
        default=0.0,
    )
    mip_gap = (
        max(0.0, (primary_value - best_bound) / primary_value)
        if primary_value > 0.0
        else 0.0
    )
    proven_optimal = (
        len(ordered) == 1
        and ordered[0].proven_optimal
        and not restrictions
    )
    node_schedules = {
        node_id: candidate.node_schedules[node_id]
        for node_id, candidate in candidates.items()
    }
    parent_candidate_id = (
        request.overlay.parent_candidate_id
        if request.overlay is not None
        else None
    )
    return SolveCandidate(
        candidate_id="vericcl-{}-{}-k{:02d}-{}".format(
            objective.value,
            backend,
            channel_count,
            candidate_cache_key(request)[:12],
        ),
        node_schedules=node_schedules,
        objective_mode=objective,
        channel_count=channel_count,
        metrics=SolverMetrics(
            status=status,
            objective_values=_objective_values(
                objective,
                makespan_us,
                resource_load_us,
                operation_count,
                operation_count,
            ),
            best_bound=best_bound,
            mip_gap=mip_gap,
            within_requested_gap=(
                all(
                    candidate.metrics.within_requested_gap
                    for candidate in ordered
                )
                and mip_gap <= request.inputs.solver.mip_gap
            ),
            solve_time_s=sum(
                candidate.metrics.solve_time_s for candidate in ordered
            ),
            model_count=sum(
                candidate.metrics.model_count for candidate in ordered
            ),
            operation_count=operation_count,
            hop_count=operation_count,
            makespan_us=makespan_us,
            maximum_normalized_resource_load=resource_load_us,
            solver_name=_common_text(
                (candidate.metrics.solver_name for candidate in ordered),
                "combined",
            ),
            solver_version=_common_text(
                (candidate.metrics.solver_version for candidate in ordered),
                request.solver_version,
            ),
            solver_seed=request.inputs.solver.solver_seed,
            thread_count=max(
                (
                    candidate.metrics.thread_count
                    for candidate in ordered
                ),
                default=0,
            ),
            termination_reason=_common_text(
                (
                    candidate.metrics.termination_reason
                    for candidate in ordered
                ),
                "combined_complete",
            ),
        ),
        selected_best=False,
        proven_optimal=proven_optimal,
        search_space_restricted=bool(restrictions),
        restrictions=tuple(sorted(restrictions)),
        parent_candidate_id=parent_candidate_id,
    )


def _constructive_channel_count(request: SolveRequest) -> int:
    if request.overlay is None or request.overlay.channel_count is None:
        return 1
    channel_count = request.overlay.channel_count
    if channel_count > request.inputs.solver.max_channels:
        raise SemanticError("overlay channel_count exceeds the solver maximum")
    return channel_count


def _effective_inputs(request: SolveRequest) -> ResolvedInput:
    if request.overlay is None or not request.overlay.temporary_forbidden:
        return request.inputs
    forbidden = set(
        request.inputs.atom_constraints.forbidden_transfers
    ) | set(request.overlay.temporary_forbidden)
    ordered = tuple(
        sorted(
            forbidden,
            key=lambda item: (
                item.slice_id,
                item.src_rank,
                item.dst_rank,
                item.stage_id,
            ),
        )
    )
    return replace(
        request.inputs,
        atom_constraints=replace(
            request.inputs.atom_constraints,
            forbidden_transfers=ordered,
        ),
    )


def _constructive_search(
    request: SolveRequest,
    problems: Tuple[SolverProblem, ...],
    objective: ObjectiveMode,
    deadline: float,
) -> tuple[
    list[SolveCandidate],
    Dict[str, SolveCandidate],
    Dict[str, Schedule],
]:
    local_constructive: Dict[str, SolveCandidate] = {}
    warm_starts: Dict[str, Schedule] = {}
    channel_count = _constructive_channel_count(request)
    if request.inputs.strategies.constructive_trees:
        for problem in problems:
            if _monotonic() >= deadline:
                break
            try:
                schedule = construct_candidate(problem, channel_count)
            except ConstructionInfeasibleError:
                continue
            candidate = _constructive_candidate(
                problem,
                schedule,
                objective,
                channel_count,
                (
                    request.overlay.parent_candidate_id
                    if request.overlay is not None
                    else None
                ),
            )
            local_constructive[problem.node.node_id] = candidate
            warm_starts[problem.node.node_id] = schedule
    global_candidates = []
    if len(local_constructive) == len(problems):
        global_candidates.append(
            _combine_node_candidates(
                request,
                local_constructive,
                "constructive",
                objective,
                channel_count,
            )
        )
    return global_candidates, local_constructive, warm_starts


def _solve_full_objective(
    request: SolveRequest,
    problems: Tuple[SolverProblem, ...],
    objective: ObjectiveMode,
    deadline: float,
) -> Tuple[SolveCandidate, ...]:
    global_candidates, _, warm_starts = _constructive_search(
        request,
        problems,
        objective,
        deadline,
    )
    if request.inputs.strategies.milp:
        base_config = request.inputs.solver
        if request.overlay is not None and request.overlay.channel_count is not None:
            base_config = replace(
                base_config,
                max_channels=request.overlay.channel_count,
            )
        models_by_node = {}
        for problem_index, problem in enumerate(problems):
            remaining = deadline - _monotonic()
            if remaining < 1.0:
                break
            remaining_nodes = len(problems) - problem_index
            node_seconds = max(1, int(remaining / remaining_nodes))
            config = replace(
                base_config,
                total_solve_timeout_s=min(
                    base_config.total_solve_timeout_s,
                    node_seconds,
                ),
                per_model_timeout_s=min(
                    base_config.per_model_timeout_s,
                    node_seconds,
                ),
            )
            try:
                models = search_models(
                    problem,
                    config,
                    objective,
                    warm_starts.get(problem.node.node_id),
                )
            except SolverUnavailableError:
                models = ()
            if (
                request.overlay is not None
                and request.overlay.channel_count is not None
            ):
                models = tuple(
                    candidate
                    for candidate in models
                    if candidate.channel_count == request.overlay.channel_count
                )
            models_by_node[problem.node.node_id] = {
                candidate.channel_count: candidate for candidate in models
            }
        common_channels = set(range(1, base_config.max_channels + 1))
        for values in models_by_node.values():
            common_channels &= set(values)
        for model_channels in sorted(common_channels):
            global_candidates.append(
                _combine_node_candidates(
                    request,
                    {
                        node_id: values[model_channels]
                        for node_id, values in models_by_node.items()
                    },
                    "milp",
                    objective,
                    model_channels,
                )
            )
    return rank_candidates(global_candidates)


def _standalone_template(unit: RoutingUnit) -> SolverTemplate:
    slice_ids = tuple(
        sorted(
            {
                slice_id
                for demand in unit.demands
                for values in (demand.contributors, demand.member_slice_ids)
                for slice_id in values
            }
        )
    )
    logical_positions = tuple(
        sorted({demand.logical_position for demand in unit.demands})
    )
    member = TemplateMember(
        unit_id=unit.unit_id,
        node_id=unit.node.node_id,
        rank_map=tuple((rank, rank) for rank in unit.node.communication_group),
        contributor_map=tuple((slice_id, slice_id) for slice_id in slice_ids),
        logical_position_map=tuple(
            (position, position) for position in logical_positions
        ),
    )
    return SolverTemplate(
        template_id="standalone-{}".format(unit.unit_id),
        representative=unit,
        members=(member,),
        exact_signature=sha256_json(
            {
                "standalone_unit_id": unit.unit_id,
                "demand_ids": tuple(
                    demand.demand_id for demand in unit.demands
                ),
            }
        ),
    )


def _merge_routing_schedules(
    base: Schedule,
    addition: Schedule,
) -> Schedule:
    if (
        base.rank_count != addition.rank_count
        or base.slice_count != addition.slice_count
        or base.slice_size_bytes != addition.slice_size_bytes
    ):
        raise SemanticError("fallback routing schedule dimensions differ")
    if (
        base.metadata.get("routing_only") is not True
        or addition.metadata.get("routing_only") is not True
    ):
        raise SemanticError("fallback merge requires routing-only schedules")
    base_ids = {transfer.transfer_id for transfer in base.transfers}
    addition_ids = {transfer.transfer_id for transfer in addition.transfers}
    if base_ids & addition_ids:
        raise SemanticError("fallback routing transfer IDs overlap")
    metadata = dict(base.metadata)
    mapping_fields = (
        "path_roots",
        "semantic_contributors",
        "semantic_predecessors",
        "tree_contributors",
        "resource_slots",
        "route_template_ids",
        "route_unit_ids",
    )
    for field in mapping_fields:
        left = metadata.get(field, {})
        right = addition.metadata.get(field, {})
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise SemanticError("routing schedule metadata is not mergeable")
        if set(left) & set(right):
            raise SemanticError("fallback routing metadata overlaps")
        metadata[field] = {**dict(left), **dict(right)}
    metadata["restrictions"] = tuple(
        sorted(
            set(metadata.get("restrictions", ()))
            | set(addition.metadata.get("restrictions", ()))
        )
    )
    return Schedule(
        schedule_id=base.schedule_id,
        transfers=tuple(
            sorted(
                base.transfers + addition.transfers,
                key=lambda transfer: transfer.transfer_id,
            )
        ),
        final_state_ids=base.final_state_ids,
        rank_count=base.rank_count,
        slice_count=base.slice_count,
        slice_size_bytes=base.slice_size_bytes,
        metadata=metadata,
    )


def _with_route_search(
    diagnostics: SearchDiagnostics,
    result: RouteSearchResult,
    *,
    fallback: bool,
) -> SearchDiagnostics:
    return replace(
        diagnostics,
        route_model_count=(
            diagnostics.route_model_count + result.launched_model_count
        ),
        fallback_member_model_count=(
            diagnostics.fallback_member_model_count
            + (result.launched_model_count if fallback else 0)
        ),
        route_model_build_time_s=(
            diagnostics.route_model_build_time_s
            + result.route_model_build_time_s
        ),
        route_model_optimize_time_s=(
            diagnostics.route_model_optimize_time_s
            + result.route_model_optimize_time_s
        ),
        maximum_variable_count=max(
            diagnostics.maximum_variable_count,
            result.maximum_variable_count,
        ),
        maximum_constraint_count=max(
            diagnostics.maximum_constraint_count,
            result.maximum_constraint_count,
        ),
        maximum_general_constraint_count=max(
            diagnostics.maximum_general_constraint_count,
            result.maximum_general_constraint_count,
        ),
    )


def _merge_objective_diagnostics(
    left: SearchDiagnostics,
    right: SearchDiagnostics,
) -> SearchDiagnostics:
    return SearchDiagnostics(
        requested_problem_count=max(
            left.requested_problem_count,
            right.requested_problem_count,
        ),
        template_count=max(left.template_count, right.template_count),
        template_member_count=max(
            left.template_member_count,
            right.template_member_count,
        ),
        route_model_count=left.route_model_count + right.route_model_count,
        fallback_member_model_count=(
            left.fallback_member_model_count
            + right.fallback_member_model_count
        ),
        route_model_build_time_s=(
            left.route_model_build_time_s
            + right.route_model_build_time_s
        ),
        route_model_optimize_time_s=(
            left.route_model_optimize_time_s
            + right.route_model_optimize_time_s
        ),
        expansion_time_s=left.expansion_time_s + right.expansion_time_s,
        scheduling_time_s=left.scheduling_time_s + right.scheduling_time_s,
        maximum_variable_count=max(
            left.maximum_variable_count,
            right.maximum_variable_count,
        ),
        maximum_constraint_count=max(
            left.maximum_constraint_count,
            right.maximum_constraint_count,
        ),
        maximum_general_constraint_count=max(
            left.maximum_general_constraint_count,
            right.maximum_general_constraint_count,
        ),
    )


def _pattern_metrics(patterns: Mapping[str, RoutePattern]) -> tuple:
    stats = [pattern.model_stats for pattern in patterns.values()]
    return (
        len(stats),
        sum(item.build_time_s for item in stats),
        sum(item.optimize_time_s for item in stats),
    )


def _scalable_candidate(
    request: SolveRequest,
    objective: ObjectiveMode,
    channel_count: int,
    node_schedules: Mapping[str, Schedule],
    global_schedule: Schedule,
    patterns: Mapping[str, RoutePattern],
    maximum_thread_count: int,
    route_cache_keys: Tuple[str, ...],
) -> SolveCandidate:
    makespan_us = _makespan(global_schedule)
    resource_load_us = _maximum_resource_load(global_schedule)
    operation_count = len(global_schedule.transfers)
    model_count, build_time, optimize_time = _pattern_metrics(patterns)
    restrictions = {
        restriction
        for schedule in node_schedules.values()
        for restriction in schedule.metadata.get("restrictions", ())
    }
    restrictions.add("template_route_composition")
    if len(request.plan.nodes) > 1:
        restrictions.add("independent_node_composition")
    identity = sha256_json(
        {
            "candidate_cache_key": candidate_cache_key(request),
            "objective": objective.value,
            "channel_count": channel_count,
            "route_keys": tuple(sorted(route_cache_keys)),
            "pattern_ids": tuple(sorted(patterns)),
        }
    )[:12]
    return SolveCandidate(
        candidate_id="vericcl-{}-template-route-k{:02d}-{}".format(
            objective.value,
            channel_count,
            identity,
        ),
        node_schedules=node_schedules,
        objective_mode=objective,
        channel_count=channel_count,
        metrics=SolverMetrics(
            status=SolveStatus.FEASIBLE,
            objective_values=_objective_values(
                objective,
                makespan_us,
                resource_load_us,
                operation_count,
                operation_count,
            ),
            best_bound=0.0,
            mip_gap=0.0,
            within_requested_gap=False,
            solve_time_s=build_time + optimize_time,
            model_count=model_count,
            operation_count=operation_count,
            hop_count=operation_count,
            makespan_us=makespan_us,
            maximum_normalized_resource_load=resource_load_us,
            solver_name="gurobi_route",
            solver_version=request.solver_version,
            solver_seed=request.inputs.solver.solver_seed,
            thread_count=maximum_thread_count,
            termination_reason="restricted_template_route_complete",
        ),
        selected_best=False,
        proven_optimal=False,
        search_space_restricted=True,
        restrictions=tuple(sorted(restrictions)),
        parent_candidate_id=(
            request.overlay.parent_candidate_id
            if request.overlay is not None
            else None
        ),
        global_schedule=global_schedule,
    )


def _solve_scalable_objective(
    request: SolveRequest,
    problems: Tuple[SolverProblem, ...],
    objective: ObjectiveMode,
    deadline: float,
) -> tuple[Tuple[SolveCandidate, ...], SearchDiagnostics]:
    from vericcl.composer import compose_routes

    global_candidates, _, _ = _constructive_search(
        request,
        problems,
        objective,
        deadline,
    )
    diagnostics = SearchDiagnostics(
        requested_problem_count=len(problems),
    )
    if not request.inputs.strategies.milp:
        return rank_candidates(global_candidates), diagnostics
    templates = build_solver_templates(
        problems,
        request.plan.planning_mode,
    )
    diagnostics = replace(
        diagnostics,
        template_count=len(templates),
        template_member_count=sum(
            len(template.members) for template in templates
        ),
    )
    base_config = request.inputs.solver
    channel_counts = None
    if request.overlay is not None and request.overlay.channel_count is not None:
        base_config = replace(
            base_config,
            max_channels=request.overlay.channel_count,
        )
        channel_counts = (request.overlay.channel_count,)
    try:
        route_search = search_route_models(
            templates,
            base_config,
            objective,
            deadline,
            channel_counts=channel_counts,
        )
    except SolverUnavailableError:
        return rank_candidates(global_candidates), diagnostics
    diagnostics = _with_route_search(
        diagnostics,
        route_search,
        fallback=False,
    )
    expected_template_ids = {
        template.template_id for template in templates
    }
    template_by_id = {
        template.template_id: template for template in templates
    }
    unit_context = {}
    for problem in problems:
        for unit in split_routing_units(problem):
            unit_context[unit.unit_id] = (unit, problem)
    for channel_count, patterns in route_search.patterns_by_channel.items():
        if set(patterns) != expected_template_ids:
            continue
        expansion_started = _diagnostic_clock()
        instantiated = instantiate_route_patterns(
            templates,
            patterns,
            problems,
        )
        expansion_elapsed = _diagnostic_clock() - expansion_started
        node_schedules = dict(instantiated.node_schedules)
        used_patterns = dict(patterns)
        route_keys = [
            route_model_cache_key(
                request.plan.planning_mode,
                template_by_id[template_id],
                objective,
                channel_count,
            )
            for template_id in sorted(patterns)
        ]
        maximum_threads = route_search.maximum_thread_count
        failures = instantiated.failures
        if failures:
            standalone_by_unit = {}
            for failure in failures:
                context = unit_context.get(failure.unit_id)
                if context is None:
                    standalone_by_unit = {}
                    break
                standalone_by_unit[failure.unit_id] = _standalone_template(
                    context[0]
                )
            if len(standalone_by_unit) != len(failures):
                continue
            standalone_templates = tuple(
                standalone_by_unit[unit_id]
                for unit_id in sorted(standalone_by_unit)
            )
            try:
                fallback_search = search_route_models(
                    standalone_templates,
                    base_config,
                    objective,
                    deadline,
                    channel_counts=(channel_count,),
                )
            except SolverUnavailableError:
                continue
            diagnostics = _with_route_search(
                diagnostics,
                fallback_search,
                fallback=True,
            )
            fallback_patterns = fallback_search.patterns_by_channel.get(
                channel_count,
                {},
            )
            expected_fallback_ids = {
                template.template_id for template in standalone_templates
            }
            if set(fallback_patterns) != expected_fallback_ids:
                continue
            maximum_threads = max(
                maximum_threads,
                fallback_search.maximum_thread_count,
            )
            fallback_failed = False
            fallback_expansion_started = _diagnostic_clock()
            for failure in failures:
                unit, problem = unit_context[failure.unit_id]
                standalone = standalone_by_unit[failure.unit_id]
                isolated = replace(
                    problem,
                    demands=unit.demands,
                    infeasible_demand_ids=tuple(
                        demand_id
                        for demand_id in problem.infeasible_demand_ids
                        if demand_id
                        in {demand.demand_id for demand in unit.demands}
                    ),
                )
                fallback_instantiation = instantiate_route_patterns(
                    (standalone,),
                    {
                        standalone.template_id: fallback_patterns[
                            standalone.template_id
                        ]
                    },
                    (isolated,),
                )
                if fallback_instantiation.failures:
                    fallback_failed = True
                    break
                node_id = problem.node.node_id
                node_schedules[node_id] = _merge_routing_schedules(
                    node_schedules[node_id],
                    fallback_instantiation.node_schedules[node_id],
                )
                used_patterns[standalone.template_id] = fallback_patterns[
                    standalone.template_id
                ]
                route_keys.append(
                    route_model_cache_key(
                        request.plan.planning_mode,
                        standalone,
                        objective,
                        channel_count,
                    )
                )
            expansion_elapsed += (
                _diagnostic_clock() - fallback_expansion_started
            )
            if fallback_failed:
                continue
        diagnostics = replace(
            diagnostics,
            expansion_time_s=(
                diagnostics.expansion_time_s + expansion_elapsed
            ),
        )
        scheduling_started = _diagnostic_clock()
        global_schedule = compose_routes(
            request.plan,
            node_schedules,
            request.topology,
            channel_count,
        )
        scheduling_elapsed = _diagnostic_clock() - scheduling_started
        diagnostics = replace(
            diagnostics,
            scheduling_time_s=(
                diagnostics.scheduling_time_s + scheduling_elapsed
            ),
        )
        global_candidates.append(
            _scalable_candidate(
                request,
                objective,
                channel_count,
                node_schedules,
                global_schedule,
                used_patterns,
                maximum_threads,
                tuple(route_keys),
            )
        )
    return rank_candidates(global_candidates), diagnostics


def _solve_objective(
    request: SolveRequest,
    problems: Tuple[SolverProblem, ...],
    objective: ObjectiveMode,
    deadline: float,
) -> tuple[Tuple[SolveCandidate, ...], SearchDiagnostics]:
    if request.inputs.solver.require_proven_optimal:
        return (
            _solve_full_objective(
                request,
                problems,
                objective,
                deadline,
            ),
            SearchDiagnostics(requested_problem_count=len(problems)),
        )
    return _solve_scalable_objective(
        request,
        problems,
        objective,
        deadline,
    )


def _relevant_calibration(
    request: SolveRequest,
    candidate: SolveCandidate,
) -> Tuple[float, bool]:
    del request, candidate
    return 0.0, True


def _auto_lower_bound(
    problems: Tuple[SolverProblem, ...],
    max_channels: int,
    deadline: float,
) -> float:
    bounds = []
    for problem_index, problem in enumerate(problems):
        remaining = deadline - _monotonic()
        if remaining < 1.0:
            break
        remaining_nodes = len(problems) - problem_index
        node_seconds = max(1, int(remaining / remaining_nodes))
        solver = replace(
            problem.inputs.solver,
            per_model_timeout_s=min(
                problem.inputs.solver.per_model_timeout_s,
                node_seconds,
            ),
        )
        configured = replace(
            problem,
            inputs=replace(problem.inputs, solver=solver),
        )
        bounds.append(
            throughput_time_lower_bound(
                configured,
                max_channels,
            ).total_us
        )
    return max(bounds, default=0.0)


def _auto_ranking_key(candidate: SolveCandidate) -> tuple:
    metrics = candidate.metrics
    return (
        metrics.makespan_us,
        metrics.operation_count,
        metrics.hop_count,
        candidate.objective_mode.value,
        candidate.candidate_id,
    )


def _select_candidate(
    candidates: Tuple[SolveCandidate, ...],
    selected: SolveCandidate,
) -> Tuple[SolveCandidate, ...]:
    return tuple(
        replace(
            candidate,
            selected_best=candidate.candidate_id == selected.candidate_id,
        )
        for candidate in candidates
    )


def _failure_status(problems: Tuple[SolverProblem, ...]) -> SolveStatus:
    if any(problem.infeasible_demand_ids for problem in problems):
        return SolveStatus.INFEASIBLE
    return SolveStatus.ERROR


def solve(
    request: SolveRequest,
    cache: Optional[CandidateCache] = None,
) -> SolveResult:
    if not isinstance(request, SolveRequest):
        raise SemanticError("solve requires a SolveRequest")
    selected_cache = _DEFAULT_CACHE if cache is None else cache
    if not isinstance(selected_cache, CandidateCache):
        raise SemanticError("cache must be a CandidateCache or None")
    try:
        normalized_plan = build_plan(request.inputs, request.topology)
    except VeriCCLError as error:
        return SolveResult(
            status=SolveStatus.ERROR,
            candidates=(),
            selected_candidate_id=None,
            cache_hit=False,
            message="plan construction failed: {}".format(error),
        )
    if normalized_plan != request.plan:
        return SolveResult(
            status=SolveStatus.ERROR,
            candidates=(),
            selected_candidate_id=None,
            cache_hit=False,
            message="request plan does not match normalized hierarchy",
        )
    cache_key = candidate_cache_key(request)
    if not request.inputs.solver.force_resolve:
        cached, cached_diagnostics = selected_cache.get_with_diagnostics(
            cache_key
        )
        if cached is not None and (
            not request.inputs.solver.require_proven_optimal
            or cached.proven_optimal
        ):
            cached = replace(cached, selected_best=True)
            return SolveResult(
                status=cached.metrics.status,
                candidates=(cached,),
                selected_candidate_id=cached.candidate_id,
                cache_hit=True,
                message="cache_hit",
                diagnostics=cached_diagnostics,
            )
    if (
        not request.inputs.strategies.constructive_trees
        and not request.inputs.strategies.milp
    ):
        return SolveResult(
            status=SolveStatus.NOT_RUN,
            candidates=(),
            selected_candidate_id=None,
            cache_hit=False,
            message="constructive and MILP backends are disabled",
        )
    if (
        request.inputs.strategies.milp
        and not request.inputs.strategies.constructive_trees
        and not GurobiAdapter.available()
    ):
        return SolveResult(
            status=SolveStatus.UNAVAILABLE,
            candidates=(),
            selected_candidate_id=None,
            cache_hit=False,
            message="MILP backend is unavailable",
        )
    diagnostics = SearchDiagnostics()
    try:
        solve_budget = float(request.inputs.solver.total_solve_timeout_s)
        if request.wall_clock_budget_s is not None:
            solve_budget = min(solve_budget, request.wall_clock_budget_s)
        deadline = _monotonic() + solve_budget
        effective_inputs = _effective_inputs(request)
        problems = tuple(
            build_solver_problem(node, effective_inputs, request.topology)
            for node in request.plan.nodes
        )
        mode = request.inputs.hyperparameters.objective_mode
        if mode is not ObjectiveMode.AUTO:
            candidates, diagnostics = _solve_objective(
                request,
                problems,
                mode,
                deadline,
            )
            message = "complete; comparison=objective_ranking"
        else:
            latency, diagnostics = _solve_objective(
                request,
                problems,
                ObjectiveMode.LATENCY,
                deadline,
            )
            if not latency:
                candidates = ()
                message = "auto latency solve produced no complete candidate"
            elif _monotonic() >= deadline:
                candidates = latency
                message = (
                    "complete; comparison=conservative_schedule_makespan; "
                    "auto_throughput=skipped; reason=budget_exhausted"
                )
            else:
                latency_best = latency[0]
                cv_relevant, calibration_stable = _relevant_calibration(
                    request,
                    latency_best,
                )
                if not math.isfinite(cv_relevant) or cv_relevant < 0.0:
                    raise SemanticError(
                        "relevant calibration CV must be finite and non-negative"
                    )
                minimum = request.inputs.hyperparameters.min_expected_improvement
                threshold = (
                    0.0
                    if minimum == 0.0
                    else max(minimum, 2.0 * cv_relevant)
                )
                lower_bound = _auto_lower_bound(
                    problems,
                    request.inputs.solver.max_channels,
                    deadline,
                )
                latency_time = latency_best.metrics.makespan_us
                gain_upper = (
                    max(0.0, (latency_time - lower_bound) / latency_time)
                    if latency_time > 0.0
                    else 0.0
                )
                if calibration_stable and gain_upper < threshold:
                    candidates = latency
                    message = (
                        "complete; comparison=conservative_schedule_makespan; "
                        "auto_throughput=skipped; gain_upper={:g}; "
                        "threshold={:g}"
                    ).format(gain_upper, threshold)
                else:
                    throughput, throughput_diagnostics = _solve_objective(
                        request,
                        problems,
                        ObjectiveMode.THROUGHPUT,
                        deadline,
                    )
                    candidates = latency + throughput
                    diagnostics = _merge_objective_diagnostics(
                        diagnostics,
                        throughput_diagnostics,
                    )
                    reason = (
                        "unstable_calibration"
                        if not calibration_stable
                        else "gain_upper_met_threshold"
                    )
                    message = (
                        "complete; comparison=conservative_schedule_makespan; "
                        "auto_throughput=solved; reason={}; gain_upper={:g}; "
                        "threshold={:g}"
                    ).format(reason, gain_upper, threshold)
    except (ConstructionInfeasibleError, SemanticError) as error:
        return SolveResult(
            status=SolveStatus.ERROR,
            candidates=(),
            selected_candidate_id=None,
            cache_hit=False,
            message="solve failed: {}".format(error),
            diagnostics=diagnostics,
        )
    if not candidates:
        return SolveResult(
            status=_failure_status(problems),
            candidates=(),
            selected_candidate_id=None,
            cache_hit=False,
            message=message,
            diagnostics=diagnostics,
        )
    if request.inputs.solver.require_proven_optimal:
        eligible = tuple(
            candidate for candidate in candidates if candidate.proven_optimal
        )
        if not eligible:
            return SolveResult(
                status=SolveStatus.ERROR,
                candidates=(),
                selected_candidate_id=None,
                cache_hit=False,
                message="optimality proof was required but not obtained",
                diagnostics=diagnostics,
            )
    else:
        eligible = candidates
    if request.inputs.hyperparameters.objective_mode is ObjectiveMode.AUTO:
        selected = min(eligible, key=_auto_ranking_key)
    else:
        selected = rank_candidates(eligible)[0]
    candidates = _select_candidate(candidates, selected)
    selected = next(
        candidate
        for candidate in candidates
        if candidate.candidate_id == selected.candidate_id
    )
    selected_cache.put(
        cache_key,
        selected,
        ttl_seconds=_CACHE_TTL_SECONDS,
        complete=True,
        diagnostics=diagnostics,
    )
    return SolveResult(
        status=selected.metrics.status,
        candidates=candidates,
        selected_candidate_id=selected.candidate_id,
        cache_hit=False,
        message=message,
        diagnostics=diagnostics,
    )
