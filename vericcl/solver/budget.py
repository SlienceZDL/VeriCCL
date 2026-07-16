import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from vericcl.errors import SemanticError


def _limit(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticError("{} must be a number".format(field_name))
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise SemanticError("{} must be finite and positive".format(field_name))
    return result


def _time(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticError("{} must be a number".format(field_name))
    result = float(value)
    if not math.isfinite(result):
        raise SemanticError("{} must be finite".format(field_name))
    return result


@dataclass(frozen=True)
class ModelBudget:
    seconds: float
    started_at: float
    deadline: float

    def __post_init__(self) -> None:
        seconds = _time(self.seconds, "model_budget.seconds")
        if seconds < 0.0:
            raise SemanticError("model_budget.seconds must be non-negative")
        started_at = _time(self.started_at, "model_budget.started_at")
        deadline = _time(self.deadline, "model_budget.deadline")
        if not math.isclose(
            deadline,
            started_at + seconds,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise SemanticError(
                "model_budget.deadline must equal started_at plus seconds"
            )
        object.__setattr__(self, "seconds", seconds)
        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "deadline", deadline)


@dataclass(frozen=True)
class SolveBudget:
    total_seconds: float
    per_model_seconds: float
    started_at: Optional[float] = None
    clock: Callable[[], float] = field(
        default=time.monotonic,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        total = _limit(self.total_seconds, "solve_budget.total_seconds")
        per_model = _limit(
            self.per_model_seconds,
            "solve_budget.per_model_seconds",
        )
        if not callable(self.clock):
            raise SemanticError("solve_budget.clock must be callable")
        started = self.clock() if self.started_at is None else self.started_at
        started = _time(started, "solve_budget.started_at")
        object.__setattr__(self, "total_seconds", total)
        object.__setattr__(self, "per_model_seconds", per_model)
        object.__setattr__(self, "started_at", started)

    @property
    def deadline(self) -> float:
        return self.started_at + self.total_seconds

    def remaining_seconds(self, now: Optional[float] = None) -> float:
        current = self.clock() if now is None else now
        current = _time(current, "solve_budget.current_time")
        return max(0.0, self.deadline - current)

    def expired(self, now: Optional[float] = None) -> bool:
        return self.remaining_seconds(now=now) <= 0.0

    def model_budget(self, now: Optional[float] = None) -> ModelBudget:
        current = self.clock() if now is None else now
        current = _time(current, "solve_budget.current_time")
        seconds = min(
            self.per_model_seconds,
            self.remaining_seconds(now=current),
        )
        return ModelBudget(
            seconds=seconds,
            started_at=current,
            deadline=current + seconds,
        )
