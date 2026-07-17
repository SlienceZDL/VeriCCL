import math
import time
from collections import defaultdict
from dataclasses import replace
from typing import Dict, Mapping, Optional, Tuple

from vericcl.composer import compose
from vericcl.errors import (
    ConstructionInfeasibleError,
    SemanticError,
    SolverUnavailableError,
    VeriCCLError,
)
from vericcl.input.models import ObjectiveMode, ResolvedInput
from vericcl.planner.build import build_plan
from vericcl.semantics.atom import Schedule
from vericcl.solver.cache import CandidateCache, candidate_cache_key
from vericcl.solver.constructive import construct_candidate
from vericcl.solver.demands import SolverProblem, build_solver_problem
from vericcl.solver.gurobi_api import GurobiAdapter
from vericcl.solver.lower_bounds import throughput_time_lower_bound
from vericcl.solver.model import (
    SolveCandidate,
    SolveRequest,
    SolveResult,
    SolveStatus,
    SolverMetrics,
)
from vericcl.solver.objectives import rank_candidates
from vericcl.solver.search import search_models


_CACHE_TTL_SECONDS = 3600.0
_DEFAULT_CACHE = CandidateCache()
_monotonic = time.monotonic


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


def _solve_objective(
    request: SolveRequest,
    problems: Tuple[SolverProblem, ...],
    objective: ObjectiveMode,
    deadline: float,
) -> Tuple[SolveCandidate, ...]:
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
        cached = selected_cache.get(cache_key)
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
            candidates = _solve_objective(
                request,
                problems,
                mode,
                deadline,
            )
            message = "complete; comparison=objective_ranking"
        else:
            latency = _solve_objective(
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
                    throughput = _solve_objective(
                        request,
                        problems,
                        ObjectiveMode.THROUGHPUT,
                        deadline,
                    )
                    candidates = latency + throughput
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
        )
    if not candidates:
        return SolveResult(
            status=_failure_status(problems),
            candidates=(),
            selected_candidate_id=None,
            cache_hit=False,
            message=message,
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
    )
    return SolveResult(
        status=selected.metrics.status,
        candidates=candidates,
        selected_candidate_id=selected.candidate_id,
        cache_hit=False,
        message=message,
    )
