import math
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from vericcl.errors import (
    ConstructionInfeasibleError,
    SemanticError,
    SolverUnavailableError,
)
from vericcl.input.models import ObjectiveMode, SolverConfig
from vericcl.semantics.atom import Schedule
from vericcl.solver.budget import ModelBudget
from vericcl.solver.demands import SolverProblem
from vericcl.solver.milp import solve_milp
from vericcl.solver.model import SolveCandidate, SolveStatus
from vericcl.solver.routing import RoutePattern
from vericcl.solver.routing_milp import solve_route_milp
from vericcl.solver.templates import SolverTemplate


_COMPLETE_STATUSES = frozenset(
    {
        SolveStatus.OPTIMAL,
        SolveStatus.FEASIBLE,
        SolveStatus.TIME_LIMIT,
    }
)
_monotonic = time.monotonic


@dataclass(frozen=True)
class RouteSearchResult:
    patterns_by_channel: Mapping[int, Mapping[str, RoutePattern]]
    launched_model_count: int
    route_model_build_time_s: float
    route_model_optimize_time_s: float
    maximum_variable_count: int
    maximum_constraint_count: int
    maximum_general_constraint_count: int
    maximum_thread_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.patterns_by_channel, Mapping):
            raise SemanticError("patterns_by_channel must be a mapping")
        normalized = {}
        for channel_count, patterns in self.patterns_by_channel.items():
            _positive_integer(channel_count, "patterns_by_channel channel")
            if not isinstance(patterns, Mapping) or not all(
                isinstance(template_id, str)
                and isinstance(pattern, RoutePattern)
                and pattern.template_id == template_id
                and pattern.channel_count == channel_count
                for template_id, pattern in patterns.items()
            ):
                raise SemanticError("patterns_by_channel contains invalid routes")
            normalized[channel_count] = MappingProxyType(
                dict(sorted(patterns.items()))
            )
        object.__setattr__(
            self,
            "patterns_by_channel",
            MappingProxyType(dict(sorted(normalized.items()))),
        )
        for field in (
            "launched_model_count",
            "maximum_variable_count",
            "maximum_constraint_count",
            "maximum_general_constraint_count",
            "maximum_thread_count",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SemanticError("{} must be a non-negative integer".format(field))
        for field in (
            "route_model_build_time_s",
            "route_model_optimize_time_s",
        ):
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0.0
            ):
                raise SemanticError("{} must be finite and non-negative".format(field))
            object.__setattr__(self, field, float(value))


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SemanticError("{} must be a positive integer".format(field))
    return value


def allocate_model_threads(
    model_count: int,
    requested_per_model: int,
    cpu_count: int,
) -> Tuple[int, ...]:
    count = _positive_integer(model_count, "model_count")
    requested = _positive_integer(
        requested_per_model,
        "requested_per_model",
    )
    cpus = _positive_integer(cpu_count, "cpu_count")
    active = min(count, cpus)
    per_model = max(1, min(requested, cpus // active))
    return tuple(per_model for _ in range(active))


def search_route_models(
    templates: tuple[SolverTemplate, ...],
    config: SolverConfig,
    objective: ObjectiveMode,
    deadline: float,
    channel_counts: Optional[Tuple[int, ...]] = None,
) -> RouteSearchResult:
    try:
        templates = tuple(templates)
    except TypeError as error:
        raise SemanticError("templates must be iterable") from error
    if not all(isinstance(template, SolverTemplate) for template in templates):
        raise SemanticError("templates must contain SolverTemplate values")
    template_ids = tuple(template.template_id for template in templates)
    if len(template_ids) != len(set(template_ids)):
        raise SemanticError("route-search template IDs must be unique")
    if not isinstance(config, SolverConfig):
        raise SemanticError("config must be a SolverConfig")
    if not isinstance(objective, ObjectiveMode) or objective is ObjectiveMode.AUTO:
        raise SemanticError("route search requires a resolved objective")
    if (
        isinstance(deadline, bool)
        or not isinstance(deadline, (int, float))
        or not math.isfinite(float(deadline))
    ):
        raise SemanticError("route search deadline must be finite")
    max_channels = _positive_integer(config.max_channels, "max_channels")
    if channel_counts is None:
        channels = tuple(range(1, max_channels + 1))
    else:
        try:
            channels = tuple(channel_counts)
        except TypeError as error:
            raise SemanticError("channel_counts must be iterable") from error
        if not channels:
            raise SemanticError("channel_counts must not be empty")
        channels = tuple(
            sorted(
                {
                    _positive_integer(value, "channel_counts")
                    for value in channels
                }
            )
        )
        if channels[-1] > max_channels:
            raise SemanticError("channel_counts exceeds config.max_channels")
    parallel_limit = _positive_integer(
        config.max_parallel_models,
        "max_parallel_models",
    )
    requested_threads = _positive_integer(
        config.max_threads_per_model,
        "max_threads_per_model",
    )
    per_model_timeout = _positive_integer(
        config.per_model_timeout_s,
        "per_model_timeout_s",
    )
    cpu_count = max(1, os.cpu_count() or 1)
    worker_limit = min(parallel_limit, cpu_count)
    jobs = tuple(
        (channel_count, template.template_id, template)
        for channel_count in channels
        for template in sorted(templates, key=lambda item: item.template_id)
    )
    patterns = {channel_count: {} for channel_count in channels}
    launched = 0
    build_time = 0.0
    optimize_time = 0.0
    maximum_variables = 0
    maximum_constraints = 0
    maximum_general_constraints = 0
    maximum_threads = 0
    backend_unavailable = False
    offset = 0
    with ThreadPoolExecutor(max_workers=worker_limit) as executor:
        while offset < len(jobs) and not backend_unavailable:
            now = _monotonic()
            if now >= deadline:
                break
            batch = jobs[offset : offset + worker_limit]
            thread_count = allocate_model_threads(
                len(batch),
                requested_threads,
                cpu_count,
            )[0]
            maximum_threads = max(maximum_threads, thread_count)
            running = {}
            for channel_count, template_id, template in batch:
                now = _monotonic()
                remaining = max(0.0, float(deadline) - now)
                if remaining <= 0.0:
                    break
                seconds = min(float(per_model_timeout), remaining)
                budget = ModelBudget(
                    seconds=seconds,
                    started_at=now,
                    deadline=now + seconds,
                )
                future = executor.submit(
                    solve_route_milp,
                    template,
                    channel_count,
                    objective,
                    budget,
                    thread_count,
                )
                running[future] = (channel_count, template_id)
                launched += 1
            if not running:
                break
            finished, _ = wait(tuple(running))
            for future in sorted(finished, key=lambda item: running[item]):
                channel_count, template_id = running[future]
                try:
                    pattern = future.result()
                except ConstructionInfeasibleError:
                    continue
                except SolverUnavailableError:
                    backend_unavailable = True
                    continue
                if not isinstance(pattern, RoutePattern):
                    raise SemanticError(
                        "route backend must return a RoutePattern"
                    )
                if (
                    pattern.template_id != template_id
                    or pattern.channel_count != channel_count
                    or pattern.objective_mode is not objective
                ):
                    raise SemanticError(
                        "route pattern does not match its search job"
                    )
                patterns[channel_count][template_id] = pattern
                stats = pattern.model_stats
                build_time += stats.build_time_s
                optimize_time += stats.optimize_time_s
                maximum_variables = max(
                    maximum_variables,
                    stats.variable_count,
                )
                maximum_constraints = max(
                    maximum_constraints,
                    stats.constraint_count,
                )
                maximum_general_constraints = max(
                    maximum_general_constraints,
                    stats.general_constraint_count,
                )
            offset += len(batch)
    return RouteSearchResult(
        patterns_by_channel=patterns,
        launched_model_count=launched,
        route_model_build_time_s=build_time,
        route_model_optimize_time_s=optimize_time,
        maximum_variable_count=maximum_variables,
        maximum_constraint_count=maximum_constraints,
        maximum_general_constraint_count=maximum_general_constraints,
        maximum_thread_count=maximum_threads,
    )


def _configured_problem(
    problem: SolverProblem,
    config: SolverConfig,
    thread_count: int,
) -> SolverProblem:
    solver = replace(config, max_threads_per_model=thread_count)
    return replace(
        problem,
        inputs=replace(problem.inputs, solver=solver),
    )


def _is_complete(candidate: SolveCandidate, problem: SolverProblem) -> bool:
    return (
        candidate.metrics.status in _COMPLETE_STATUSES
        and set(candidate.node_schedules) == {problem.node.node_id}
    )


def search_models(
    problem: SolverProblem,
    config: SolverConfig,
    objective: ObjectiveMode,
    warm_start: Optional[Schedule],
) -> Tuple[SolveCandidate, ...]:
    if not isinstance(problem, SolverProblem):
        raise SemanticError("problem must be a SolverProblem")
    if not isinstance(config, SolverConfig):
        raise SemanticError("config must be a SolverConfig")
    if not isinstance(objective, ObjectiveMode):
        raise SemanticError("objective must be an ObjectiveMode")
    if objective is ObjectiveMode.AUTO:
        raise SemanticError("AUTO must be resolved before model search")
    if warm_start is not None and not isinstance(warm_start, Schedule):
        raise SemanticError("warm_start must be a Schedule or None")
    if config.max_channels > problem.inputs.solver.max_channels:
        raise SemanticError(
            "config.max_channels exceeds the built candidate edge range"
        )
    max_channels = _positive_integer(config.max_channels, "max_channels")
    parallel_limit = _positive_integer(
        config.max_parallel_models,
        "max_parallel_models",
    )
    requested_threads = _positive_integer(
        config.max_threads_per_model,
        "max_threads_per_model",
    )
    total_timeout = _positive_integer(
        config.total_solve_timeout_s,
        "total_solve_timeout_s",
    )
    per_model_timeout = _positive_integer(
        config.per_model_timeout_s,
        "per_model_timeout_s",
    )
    cpu_count = max(1, os.cpu_count() or 1)
    worker_count = min(max_channels, parallel_limit, cpu_count)
    thread_count = allocate_model_threads(
        worker_count,
        requested_threads,
        cpu_count,
    )[0]
    configured_problem = _configured_problem(
        problem,
        config,
        thread_count,
    )
    started_at = _monotonic()
    deadline = started_at + total_timeout
    next_channel = 1
    running = {}
    completed = []
    launched_count = 0
    backend_unavailable = False

    def submit_one(executor) -> bool:
        nonlocal next_channel, launched_count
        if backend_unavailable or next_channel > max_channels:
            return False
        now = _monotonic()
        remaining = max(0.0, deadline - now)
        if remaining <= 0.0:
            return False
        seconds = min(float(per_model_timeout), remaining)
        budget = ModelBudget(
            seconds=seconds,
            started_at=now,
            deadline=now + seconds,
        )
        channel_count = next_channel
        model_index = launched_count
        next_channel += 1
        launched_count += 1
        future = executor.submit(
            solve_milp,
            configured_problem,
            channel_count,
            objective,
            budget,
            warm_start,
        )
        running[future] = (model_index, channel_count)
        return True

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for _ in range(worker_count):
            if not submit_one(executor):
                break
        while running:
            finished, _ = wait(tuple(running), return_when=FIRST_COMPLETED)
            ordered = sorted(finished, key=lambda item: running[item])
            for future in ordered:
                model_index, channel_count = running.pop(future)
                try:
                    candidate = future.result()
                except SolverUnavailableError:
                    backend_unavailable = True
                    continue
                if not isinstance(candidate, SolveCandidate):
                    raise SemanticError(
                        "MILP backend must return a SolveCandidate"
                    )
                if candidate.channel_count != channel_count:
                    raise SemanticError(
                        "MILP candidate channel count does not match its model"
                    )
                if _is_complete(candidate, configured_problem):
                    completed.append((model_index, candidate))
            while len(running) < worker_count and submit_one(executor):
                pass
    return tuple(
        replace(
            candidate,
            metrics=replace(
                candidate.metrics,
                model_count=launched_count,
                model_index=model_index,
            ),
        )
        for model_index, candidate in sorted(
            completed,
            key=lambda item: item[1].channel_count,
        )
    )
