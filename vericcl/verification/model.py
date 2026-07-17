from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from vericcl.errors import SemanticError


class ValidationStatus(str, Enum):
    NOT_RUN = "not_run"
    VALID = "valid"
    INVALID = "invalid"
    FATAL = "fatal"
    WARNING = "warning"
    ANALYSIS_ERROR = "analysis_error"
    FAILED = "failed"


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SemanticError("{} must be a non-empty string".format(field))
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class CheckResult:
    dimension: str
    status: ValidationStatus
    code: str
    message: str
    evidence: Mapping[str, object]

    def __post_init__(self) -> None:
        _identifier(self.dimension, "check_result.dimension")
        if not isinstance(self.status, ValidationStatus):
            raise SemanticError(
                "check_result.status must be a ValidationStatus"
            )
        _identifier(self.code, "check_result.code")
        _identifier(self.message, "check_result.message")
        if not isinstance(self.evidence, Mapping):
            raise SemanticError("check_result.evidence must be a mapping")
        object.__setattr__(self, "evidence", _freeze(self.evidence))


_DIMENSIONS = (
    "input",
    "semantic",
    "state",
    "topology",
    "timing",
    "resource",
    "buffer",
    "endpoint",
    "deadlock",
    "xml",
    "bdd",
    "simulation",
    "runtime",
    "online",
)

_CORE_DIMENSIONS = (
    "input",
    "semantic",
    "state",
    "topology",
    "timing",
    "resource",
    "buffer",
    "endpoint",
    "deadlock",
    "xml",
    "simulation",
)


@dataclass(frozen=True)
class ValidationReport:
    input: CheckResult
    semantic: CheckResult
    state: CheckResult
    topology: CheckResult
    timing: CheckResult
    resource: CheckResult
    buffer: CheckResult
    endpoint: CheckResult
    deadlock: CheckResult
    xml: CheckResult
    bdd: CheckResult
    simulation: CheckResult
    runtime: CheckResult
    online: CheckResult

    def __post_init__(self) -> None:
        for dimension in _DIMENSIONS:
            result = getattr(self, dimension)
            if not isinstance(result, CheckResult):
                raise SemanticError(
                    "validation report {} result is invalid".format(
                        dimension
                    )
                )
            if result.dimension != dimension:
                raise SemanticError(
                    "validation report {} dimension does not match".format(
                        dimension
                    )
                )

    @property
    def overall_status(self) -> ValidationStatus:
        statuses = tuple(
            getattr(self, dimension).status
            for dimension in _CORE_DIMENSIONS
        )
        if ValidationStatus.FATAL in statuses:
            return ValidationStatus.FATAL
        if ValidationStatus.INVALID in statuses:
            return ValidationStatus.INVALID
        if ValidationStatus.FAILED in statuses:
            return ValidationStatus.FAILED
        if ValidationStatus.NOT_RUN in statuses:
            return ValidationStatus.NOT_RUN
        return ValidationStatus.VALID

    @property
    def runtime_compatible(self) -> bool:
        return self.runtime.status is ValidationStatus.VALID

    @property
    def eligible_for_selection(self) -> bool:
        return (
            self.overall_status is ValidationStatus.VALID
            and self.bdd.status is ValidationStatus.VALID
            and self.simulation.status is ValidationStatus.VALID
            and self.runtime_compatible
            and self.online.status
            in {ValidationStatus.VALID, ValidationStatus.NOT_RUN}
        )

    @property
    def online_validated(self) -> bool:
        return self.online.status is ValidationStatus.VALID
