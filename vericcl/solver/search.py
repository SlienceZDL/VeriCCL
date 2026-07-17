import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import replace
from typing import Optional, Tuple

from vericcl.errors import SemanticError, SolverUnavailableError
from vericcl.input.models import ObjectiveMode, SolverConfig
from vericcl.semantics.atom import Schedule
from vericcl.solver.budget import ModelBudget
from vericcl.solver.demands import SolverProblem
from vericcl.solver.milp import solve_milp
from vericcl.solver.model import SolveCandidate, SolveStatus


_COMPLETE_STATUSES = frozenset(
    {
        SolveStatus.OPTIMAL,
        SolveStatus.FEASIBLE,
        SolveStatus.TIME_LIMIT,
    }
)
_monotonic = time.monotonic


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
