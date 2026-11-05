"""Independent evidence dimensions and their public command exit codes."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class ExitCode(IntEnum):
    SUCCESS = 0
    INPUT = 2
    OFFLINE = 3
    ONLINE = 4
    INTERNAL = 5
    INVALID = 6
    STORAGE = 7
    BACKEND = 8
    CANCELLED = 130


@dataclass(frozen=True)
class RunResult:
    contract_met: bool
    execution: str
    checks: dict[str, str] = field(default_factory=dict)
    model_coverage: str = "unknown"
    acceptance: str = "not_evaluated"
    stop_reason: str = "unspecified"
    proof_scope: dict | None = None
    optional_failures: tuple[str, ...] = ()
    failure_category: str | None = None

    def __post_init__(self):
        if self.execution not in {"completed", "incomplete", "failed", "cancelled", "not_started"}:
            raise ValueError("invalid execution status")
        if self.contract_met:
            if self.execution != "completed" or self.failure_category is not None:
                raise ValueError("completed contract requires completed execution")
            if any(v not in {"valid", "not_required"} for v in self.checks.values()):
                raise ValueError("required checks must be valid for a completed contract")
        elif self.failure_category not in {name.lower() for name in ExitCode.__members__ if name != "SUCCESS"}:
            raise ValueError("incomplete contract requires a failure category")

    @property
    def exit_code(self) -> int:
        return int(ExitCode.SUCCESS if self.contract_met else ExitCode[self.failure_category.upper()])


@dataclass(frozen=True)
class FinalizationResult:
    exit_code: int
    complete: bool
    errors: tuple[str, ...] = ()
