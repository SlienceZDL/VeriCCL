import math
import os
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from queue import Queue
from typing import Dict, Mapping, Optional, Tuple

from vericcl.errors import (
    ConstructionInfeasibleError,
    SemanticError,
    SolverUnavailableError,
)
from vericcl.input.models import ObjectiveMode, ResolvedInput
from vericcl.semantics.atom import Schedule
from vericcl.solver.budget import ModelBudget
from vericcl.solver.cache import build_cache_signature, candidate_cache_key
from vericcl.solver.constructive import construct_route_pattern
from vericcl.solver.demands import SolverProblem
from vericcl.solver.gurobi_api import GurobiAdapter
from vericcl.solver.instantiate import instantiate_route_patterns
from vericcl.solver.model import (
    SearchDiagnostics,
    SolveCandidate,
    SolveRequest,
    SolveStatus,
    SolverMetrics,
)
from vericcl.solver.objectives import rank_candidates
from vericcl.solver.routing import RoutePattern
from vericcl.solver.routing_milp import solve_route_milp
from vericcl.solver.search import allocate_model_threads
from vericcl.solver.templates import (
    RoutingUnit,
    SolverTemplate,
    TemplateMember,
    build_solver_templates,
    split_routing_units,
)
from vericcl.topology.model import LinkKey, Topology


_monotonic = time.monotonic


@dataclass(frozen=True)
class TemplateSearchResult:
    candidates: Tuple[SolveCandidate, ...]
    diagnostics: SearchDiagnostics

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        if not all(isinstance(item, SolveCandidate) for item in candidates):
            raise SemanticError(
                "template search candidates must contain SolveCandidate values"
            )
        if not isinstance(self.diagnostics, SearchDiagnostics):
            raise SemanticError(
                "template search diagnostics must be SearchDiagnostics"
            )
        object.__setattr__(self, "candidates", candidates)


@dataclass(frozen=True, order=True)
class _WorkItem:
    objective_value: str
    channel_count: int
    template_id: str
    template: SolverTemplate

    def identity(self) -> str:
        return "objective={}, channel_count={}, template_id={}".format(
            self.objective_value,
            self.channel_count,
            self.template_id,
        )


@dataclass
class _Measurements:
    route_model_count: int = 0
    fallback_member_model_count: int = 0
    build_time_s: float = 0.0
    optimize_time_s: float = 0.0
    expansion_time_s: float = 0.0
    scheduling_time_s: float = 0.0
    variables_max: int = 0
    constraints_max: int = 0
    general_constraints_max: int = 0

    def record_pattern(self, pattern: RoutePattern) -> None:
        stats = pattern.model_stats
        self.build_time_s += stats.build_time_s
        self.optimize_time_s += stats.optimize_time_s
        self.variables_max = max(self.variables_max, stats.variable_count)
        self.constraints_max = max(
            self.constraints_max,
            stats.constraint_count,
        )
        self.general_constraints_max = max(
            self.general_constraints_max,
            stats.general_constraint_count,
        )


def _validate_api(
    request: SolveRequest,
    problems: Tuple[SolverProblem, ...],
    objective: ObjectiveMode,
    deadline: float,
) -> Tuple[SolverProblem, ...]:
    if not isinstance(request, SolveRequest):
        raise SemanticError("request must be a SolveRequest")
    try:
        values = tuple(problems)
    except TypeError as error:
        raise SemanticError("problems must be iterable") from error
    if not values or not all(
        isinstance(problem, SolverProblem) for problem in values
    ):
        raise SemanticError("problems must contain SolverProblem values")
    if not isinstance(objective, ObjectiveMode):
        raise SemanticError("objective must be an ObjectiveMode")
    if objective is ObjectiveMode.AUTO:
        raise SemanticError("AUTO must be resolved before template search")
    if (
        isinstance(deadline, bool)
        or not isinstance(deadline, (int, float))
        or not math.isfinite(float(deadline))
    ):
        raise SemanticError("deadline must be finite")
    expected = {node.node_id for node in request.plan.nodes}
    if {problem.node.node_id for problem in values} != expected:
        raise SemanticError("one solver problem is required for every plan node")
    if any(problem.topology != request.topology for problem in values):
        raise SemanticError("solver problem topology does not match request")
    first = values[0].inputs
    if any(problem.inputs != first for problem in values):
        raise SemanticError("solver problems must use one resolved input")
    return values


def _configured_inputs(inputs: ResolvedInput, thread_count: int) -> ResolvedInput:
    return replace(
        inputs,
        solver=replace(
            inputs.solver,
            max_threads_per_model=thread_count,
        ),
    )


def _complete_status(patterns: Tuple[RoutePattern, ...]) -> SolveStatus:
    statuses = {pattern.metrics.status for pattern in patterns}
    if statuses == {SolveStatus.OPTIMAL}:
        return SolveStatus.OPTIMAL
    if SolveStatus.TIME_LIMIT in statuses:
        return SolveStatus.TIME_LIMIT
    return SolveStatus.FEASIBLE


def _makespan(schedule: Schedule) -> float:
    return max(
        (transfer.ed_time for transfer in schedule.transfers),
        default=0.0,
    )


def _maximum_resource_load(
    schedule: Schedule,
    topology: Topology,
    channel_count: int,
) -> float:
    if (
        isinstance(channel_count, bool)
        or not isinstance(channel_count, int)
        or channel_count < 1
    ):
        raise SemanticError("channel_count must be a positive integer")
    raw_slots = schedule.metadata.get("resource_slots", {})
    if not isinstance(raw_slots, Mapping):
        raise SemanticError("resource_slots metadata must be a mapping")
    loads = defaultdict(float)
    for transfer in schedule.transfers:
        duration = transfer.ed_time - transfer.st_time
        physical = LinkKey(transfer.src_rank, transfer.dst_rank)
        edge = topology.link(physical)
        loads[("link", physical)] += duration
        slots = raw_slots.get(transfer.transfer_id, {})
        if not isinstance(slots, Mapping):
            raise SemanticError("transfer resource slots must be a mapping")
        if set(slots) != set(edge.resource_ids):
            raise SemanticError(
                "transfer resource slots do not cover topology resources"
            )
        for resource_id in slots:
            resource = topology.shared_resources.get(resource_id)
            if resource is None or physical not in resource.member_links:
                raise SemanticError(
                    "transfer resource slot does not match the topology"
                )
            loads[("resource", resource_id)] += duration
    normalized = []
    for key, load in loads.items():
        if key[0] == "link":
            capacity = topology.link(key[1]).max_channels
        else:
            capacity = topology.shared_resources[key[1]].max_channels
        normalized.append(load / min(channel_count, capacity))
    return max(normalized, default=0.0)


def _instantiated_hop_count(schedule: Schedule) -> int:
    value = schedule.metadata.get("instantiated_path_hop_count")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SemanticError(
            "global route schedule must report instantiated path hops"
        )
    return value


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


def _common_text(values, fallback: str) -> str:
    unique = sorted(set(values))
    return unique[0] if len(unique) == 1 else fallback


def _fallback_template(
    parent: SolverTemplate,
    unit: RoutingUnit,
) -> SolverTemplate:
    contributor_ids = set()
    position_ids = set()
    for demand in unit.demands:
        contributor_ids.update(demand.contributors)
        contributor_ids.update(demand.member_slice_ids)
        contributor_ids.update(
            item.slice_id for item in demand.forbidden_members
        )
        position_ids.add(demand.logical_position)
    identity = TemplateMember(
        unit_id=unit.unit_id,
        node_id=unit.node.node_id,
        rank_map=tuple((rank, rank) for rank in unit.node.communication_group),
        contributor_map=tuple(
            (value, value) for value in sorted(contributor_ids)
        ),
        logical_position_map=tuple(
            (value, value) for value in sorted(position_ids)
        ),
    )
    return SolverTemplate(
        template_id="{}-fallback-{}".format(
            parent.template_id,
            unit.unit_id,
        ),
        representative=unit,
        members=(identity,),
        exact_signature="{}-fallback-{}".format(
            parent.exact_signature,
            unit.unit_id,
        ),
    )


def _candidate(
    request: SolveRequest,
    objective: ObjectiveMode,
    channel_count: int,
    patterns: Tuple[RoutePattern, ...],
    node_schedules: Mapping[str, Schedule],
    global_schedule: Schedule,
    model_count: int,
    cache_key: Optional[str] = None,
) -> SolveCandidate:
    makespan_us = _makespan(global_schedule)
    resource_load_us = _maximum_resource_load(
        global_schedule,
        request.topology,
        channel_count,
    )
    operation_count = len(global_schedule.transfers)
    hop_count = _instantiated_hop_count(global_schedule)
    restrictions = {
        restriction
        for schedule in node_schedules.values()
        for restriction in schedule.metadata.get("restrictions", ())
    }
    restrictions.add("template_route_composition")
    if len(request.plan.nodes) > 1:
        restrictions.add("independent_node_composition")
    primary = (
        makespan_us
        if objective is ObjectiveMode.LATENCY
        else resource_load_us
    )
    mip_gap = 1.0 if primary > 0.0 else 0.0
    parent_id = (
        request.overlay.parent_candidate_id
        if request.overlay is not None
        else None
    )
    identity_key = (
        candidate_cache_key(request) if cache_key is None else cache_key
    )
    return SolveCandidate(
        candidate_id="vericcl-{}-template-k{:02d}-{}".format(
            objective.value,
            channel_count,
            identity_key[:12],
        ),
        node_schedules=node_schedules,
        objective_mode=objective,
        channel_count=channel_count,
        metrics=SolverMetrics(
            status=_complete_status(patterns),
            objective_values=_objective_values(
                objective,
                makespan_us,
                resource_load_us,
                operation_count,
                hop_count,
            ),
            best_bound=0.0,
            mip_gap=mip_gap,
            within_requested_gap=False,
            solve_time_s=0.0,
            model_count=model_count,
            operation_count=operation_count,
            hop_count=hop_count,
            makespan_us=makespan_us,
            maximum_normalized_resource_load=resource_load_us,
            solver_name=_common_text(
                (pattern.metrics.solver_name for pattern in patterns),
                "combined-route",
            ),
            solver_version=_common_text(
                (pattern.metrics.solver_version for pattern in patterns),
                request.solver_version,
            ),
            solver_seed=request.inputs.solver.solver_seed,
            thread_count=max(
                (pattern.metrics.thread_count for pattern in patterns),
                default=0,
            ),
            termination_reason="template_route_composition_complete",
        ),
        selected_best=False,
        proven_optimal=False,
        search_space_restricted=True,
        restrictions=tuple(sorted(restrictions)),
        parent_candidate_id=parent_id,
        global_schedule=global_schedule,
    )


def _work_failure(item: _WorkItem, error: Exception) -> SemanticError:
    return SemanticError(
        "template route work item failed ({}): {}".format(
            item.identity(),
            error,
        )
    )


def _requires_explicit_environment() -> bool:
    return bool(
        getattr(
            solve_route_milp,
            "requires_explicit_environment",
            False,
        )
    )


def search_route_models(
    request: SolveRequest,
    problems: Tuple[SolverProblem, ...],
    objective: ObjectiveMode,
    deadline: float,
) -> TemplateSearchResult:
    from vericcl.composer import compose_routes

    problems = _validate_api(request, problems, objective, deadline)
    inputs = problems[0].inputs
    templates = build_solver_templates(
        problems,
        request.plan.planning_mode,
    )
    cache_key = candidate_cache_key(
        request,
        build_cache_signature(request, problems, templates),
    )
    units = tuple(
        unit for problem in problems for unit in split_routing_units(problem)
    )
    measurements = _Measurements()
    if not templates:
        return TemplateSearchResult(
            (),
            SearchDiagnostics(
                requested_problem_count=len(problems),
                routing_unit_count=len(units),
            ),
        )
    max_channels = inputs.solver.max_channels
    channel_counts = (
        (request.overlay.channel_count,)
        if request.overlay is not None
        and request.overlay.channel_count is not None
        else tuple(range(1, max_channels + 1))
    )
    work_items = tuple(
        _WorkItem(
            objective.value,
            channel_count,
            template.template_id,
            template,
        )
        for channel_count in channel_counts
        for template in sorted(templates, key=lambda item: item.template_id)
    )
    patterns: Dict[Tuple[int, str], RoutePattern] = {}
    next_work = 0
    backend_unavailable = False

    if inputs.strategies.milp:
        cpu_count = max(1, os.cpu_count() or 1)
        worker_count = min(
            len(work_items),
            inputs.solver.max_parallel_models,
            cpu_count,
        )
        thread_count = allocate_model_threads(
            worker_count,
            inputs.solver.max_threads_per_model,
            cpu_count,
        )[0]
        configured_inputs = _configured_inputs(inputs, thread_count)
        running = {}
        environments = []
        environment_pool = None
        if _requires_explicit_environment() and _monotonic() < deadline:
            environment_pool = Queue()
            try:
                for index in range(worker_count):
                    environment = GurobiAdapter.create_environment()
                    environments.append(environment)
                    environment_pool.put((index, environment))
            except SolverUnavailableError:
                backend_unavailable = True
                for environment in environments:
                    GurobiAdapter.dispose_environment(environment)
                environments = []
                environment_pool = None

        def solve_one(item: _WorkItem, budget: ModelBudget):
            if environment_pool is None:
                return solve_route_milp(
                    item.template,
                    configured_inputs,
                    request.topology,
                    item.channel_count,
                    objective,
                    budget,
                    None,
                )
            environment_index, environment = environment_pool.get()
            try:
                return solve_route_milp(
                    item.template,
                    configured_inputs,
                    request.topology,
                    item.channel_count,
                    objective,
                    budget,
                    None,
                    environment=environment,
                )
            finally:
                environment_pool.put((environment_index, environment))

        def submit_one(executor) -> bool:
            nonlocal next_work
            if backend_unavailable or next_work >= len(work_items):
                return False
            now = _monotonic()
            remaining = max(0.0, float(deadline) - now)
            if remaining <= 0.0:
                return False
            seconds = min(
                float(inputs.solver.per_model_timeout_s),
                remaining,
            )
            budget = ModelBudget(seconds, now, now + seconds)
            item = work_items[next_work]
            next_work += 1
            measurements.route_model_count += 1
            future = executor.submit(solve_one, item, budget)
            running[future] = item
            return True

        try:
            if not backend_unavailable:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    for _ in range(worker_count):
                        if not submit_one(executor):
                            break
                    while running:
                        finished, _ = wait(
                            tuple(running),
                            return_when=FIRST_COMPLETED,
                        )
                        for future in sorted(
                            finished,
                            key=lambda value: running[value],
                        ):
                            item = running.pop(future)
                            try:
                                pattern = future.result()
                            except SolverUnavailableError:
                                backend_unavailable = True
                                pattern = None
                            except ConstructionInfeasibleError:
                                pattern = None
                            except Exception as error:
                                raise _work_failure(item, error) from error
                            if (
                                pattern is None
                                and inputs.strategies.constructive_trees
                            ):
                                try:
                                    pattern = construct_route_pattern(
                                        item.template,
                                        configured_inputs,
                                        request.topology,
                                        item.channel_count,
                                        objective,
                                    )
                                except ConstructionInfeasibleError:
                                    pattern = None
                            if pattern is not None:
                                if (
                                    not isinstance(pattern, RoutePattern)
                                    or pattern.template_id != item.template_id
                                    or pattern.channel_count
                                    != item.channel_count
                                    or pattern.objective_mode is not objective
                                ):
                                    raise SemanticError(
                                        "route backend returned a mismatched pattern"
                                    )
                                patterns[
                                    (item.channel_count, item.template_id)
                                ] = pattern
                                if pattern.metrics.model_count > 0:
                                    measurements.record_pattern(pattern)
                        while (
                            len(running) < worker_count
                            and submit_one(executor)
                        ):
                            pass
        finally:
            for environment in environments:
                GurobiAdapter.dispose_environment(environment)
    if not inputs.strategies.milp or backend_unavailable:
        configured_inputs = inputs
        if inputs.strategies.constructive_trees:
            for item in work_items[next_work:]:
                if _monotonic() >= deadline:
                    break
                try:
                    pattern = construct_route_pattern(
                        item.template,
                        configured_inputs,
                        request.topology,
                        item.channel_count,
                        objective,
                    )
                except ConstructionInfeasibleError:
                    continue
                patterns[(item.channel_count, item.template_id)] = pattern

    units_by_key = {
        (unit.node.node_id, unit.unit_id): unit for unit in units
    }
    candidates = []
    for channel_count in channel_counts:
        base_patterns = {
            template.template_id: patterns[(channel_count, template.template_id)]
            for template in templates
            if (channel_count, template.template_id) in patterns
        }
        if len(base_patterns) != len(templates):
            continue
        expansion_started = _monotonic()
        instantiated = instantiate_route_patterns(
            request.plan,
            templates,
            base_patterns,
            inputs,
            request.topology,
        )
        expansion_time = max(0.0, _monotonic() - expansion_started)
        candidate_templates = list(templates)
        candidate_patterns = dict(base_patterns)
        if instantiated.failures:
            failed_keys = {
                (failure.node_id, failure.unit_id)
                for failure in instantiated.failures
            }
            fallback_templates = []
            fallback_patterns = {}
            incomplete = False
            for failure_key in sorted(failed_keys):
                unit = units_by_key.get(failure_key)
                parent = next(
                    (
                        template
                        for template in templates
                        if any(
                            (member.node_id, member.unit_id) == failure_key
                            for member in template.members
                        )
                    ),
                    None,
                )
                if unit is None or parent is None:
                    incomplete = True
                    break
                if parent.representative.unit_id == unit.unit_id:
                    incomplete = True
                    break
                fallback = _fallback_template(parent, unit)
                pattern = None
                if inputs.strategies.milp:
                    now = _monotonic()
                    remaining = max(0.0, float(deadline) - now)
                    if remaining > 0.0:
                        seconds = min(
                            float(inputs.solver.per_model_timeout_s),
                            remaining,
                        )
                        budget = ModelBudget(seconds, now, now + seconds)
                        cpu_count = max(1, os.cpu_count() or 1)
                        fallback_inputs = _configured_inputs(
                            inputs,
                            min(
                                inputs.solver.max_threads_per_model,
                                cpu_count,
                            ),
                        )
                        item = _WorkItem(
                            objective.value,
                            channel_count,
                            fallback.template_id,
                            fallback,
                        )
                        environment = None
                        try:
                            if _requires_explicit_environment():
                                environment = (
                                    GurobiAdapter.create_environment()
                                )
                            measurements.fallback_member_model_count += 1
                            if environment is None:
                                pattern = solve_route_milp(
                                    fallback,
                                    fallback_inputs,
                                    request.topology,
                                    channel_count,
                                    objective,
                                    budget,
                                    None,
                                )
                            else:
                                pattern = solve_route_milp(
                                    fallback,
                                    fallback_inputs,
                                    request.topology,
                                    channel_count,
                                    objective,
                                    budget,
                                    None,
                                    environment=environment,
                                )
                        except (
                            ConstructionInfeasibleError,
                            SolverUnavailableError,
                        ):
                            pattern = None
                        except Exception as error:
                            raise _work_failure(item, error) from error
                        finally:
                            if environment is not None:
                                GurobiAdapter.dispose_environment(
                                    environment
                                )
                        if pattern is not None:
                            measurements.record_pattern(pattern)
                if pattern is None and inputs.strategies.constructive_trees:
                    try:
                        pattern = construct_route_pattern(
                            fallback,
                            inputs,
                            request.topology,
                            channel_count,
                            objective,
                        )
                    except ConstructionInfeasibleError:
                        pattern = None
                if pattern is None:
                    incomplete = True
                    break
                fallback_templates.append(fallback)
                fallback_patterns[fallback.template_id] = pattern
            if incomplete:
                continue
            adjusted = []
            for template in templates:
                members = tuple(
                    member
                    for member in template.members
                    if (member.node_id, member.unit_id) not in failed_keys
                )
                if not members:
                    candidate_patterns.pop(template.template_id, None)
                    continue
                if template.representative.unit_id not in {
                    member.unit_id for member in members
                }:
                    incomplete = True
                    break
                adjusted.append(replace(template, members=members))
            if incomplete:
                continue
            candidate_templates = adjusted + fallback_templates
            candidate_patterns.update(fallback_patterns)
            expansion_started = _monotonic()
            instantiated = instantiate_route_patterns(
                request.plan,
                tuple(candidate_templates),
                candidate_patterns,
                inputs,
                request.topology,
            )
            expansion_time += max(
                0.0,
                _monotonic() - expansion_started,
            )
            if instantiated.failures:
                continue
        scheduling_started = _monotonic()
        global_schedule = compose_routes(
            request.plan,
            instantiated.node_schedules,
            request.topology,
            channel_count,
        )
        scheduling_time = max(0.0, _monotonic() - scheduling_started)
        candidate_pattern_values = tuple(
            candidate_patterns[template.template_id]
            for template in sorted(
                candidate_templates,
                key=lambda item: item.template_id,
            )
        )
        candidates.append(
            _candidate(
                request,
                objective,
                channel_count,
                candidate_pattern_values,
                instantiated.node_schedules,
                global_schedule,
                sum(
                    pattern.metrics.model_count
                    for pattern in candidate_pattern_values
                ),
                cache_key,
            )
        )
        measurements.expansion_time_s += expansion_time
        measurements.scheduling_time_s += scheduling_time
    diagnostics = SearchDiagnostics(
        requested_problem_count=len(problems),
        routing_unit_count=len(units),
        template_count=len(templates),
        template_member_count=sum(
            len(template.members) for template in templates
        ),
        route_model_count=measurements.route_model_count,
        fallback_member_model_count=(
            measurements.fallback_member_model_count
        ),
        search_model_count_total=(
            measurements.route_model_count
            + measurements.fallback_member_model_count
        ),
        route_model_build_time_s=measurements.build_time_s,
        route_model_optimize_time_s=measurements.optimize_time_s,
        template_expansion_time_s=measurements.expansion_time_s,
        global_scheduling_time_s=measurements.scheduling_time_s,
        model_variables_max=measurements.variables_max,
        model_constraints_max=measurements.constraints_max,
        model_general_constraints_max=measurements.general_constraints_max,
    )
    return TemplateSearchResult(
        rank_candidates(candidates),
        diagnostics,
    )
