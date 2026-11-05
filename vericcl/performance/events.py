from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from vericcl.errors import SemanticError
from vericcl.performance.model import finite


@dataclass(frozen=True)
class WorkOperation:
    operation_id: str
    kind: str
    cost_key: str | None
    payload_bytes: int
    dependencies: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    flow_id: str | None = None
    not_before_us: float = 0.0
    source_override: str | None = None

    def __post_init__(self):
        if not isinstance(self.operation_id, str) or not self.operation_id:
            raise SemanticError("work operation needs an identity")
        if self.source_override not in {None, 'estimated'}:
            raise SemanticError("operation source may only downgrade a proxy cost to estimated")
        if self.kind not in {"send", "rrc", "copy", "reduce", "nop"}:
            raise SemanticError("unknown physical work kind")
        if isinstance(self.payload_bytes, bool) or not isinstance(self.payload_bytes, int) or self.payload_bytes < 0:
            raise SemanticError("payload must be a nonnegative integer byte count")
        if self.kind == "nop":
            if self.payload_bytes or self.cost_key is not None:
                raise SemanticError("NOP carries no data or service cost")
        elif self.payload_bytes == 0 or not isinstance(self.cost_key, str) or not self.cost_key:
            raise SemanticError("data operation needs payload and a cost component")
        if self.kind in {"send", "rrc"} and (not isinstance(self.flow_id, str) or not self.flow_id):
            raise SemanticError("communication needs a semantic flow identity")
        if self.kind in {"copy", "reduce", "nop"} and self.resources:
            raise SemanticError("independent local operations do not use mixed communication resources")
        for name in ("dependencies", "resources"):
            values = tuple(getattr(self, name))
            if any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
                raise SemanticError("invalid operation " + name)
            object.__setattr__(self, name, values)
        object.__setattr__(self, "not_before_us", finite(self.not_before_us, "hard release time"))


@dataclass(frozen=True)
class Executable:
    operations: tuple[WorkOperation, ...]
    output_operations: tuple[str, ...] | None = None
    context: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        operations = tuple(self.operations)
        if not all(isinstance(operation, WorkOperation) for operation in operations):
            raise SemanticError("executable operations must be physical work records")
        ids = {op.operation_id for op in operations}
        if len(ids) != len(operations):
            raise SemanticError("duplicate physical operation")
        if any(set(op.dependencies) - ids for op in operations):
            raise SemanticError("unknown operation dependency")
        outputs = tuple(ids) if self.output_operations is None else tuple(self.output_operations)
        if set(outputs) - ids:
            raise SemanticError("unknown output operation")
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "output_operations", tuple(sorted(outputs)))


@dataclass(frozen=True)
class Endpoint:
    role: str
    operation: WorkOperation


def pair_endpoints(endpoints) -> Executable:
    grouped = {}
    for endpoint in endpoints:
        if not isinstance(endpoint, Endpoint):
            raise SemanticError("invalid endpoint")
        grouped.setdefault(endpoint.operation.operation_id, []).append(endpoint)
    operations = []
    for members in grouped.values():
        operation = members[0].operation
        required = {"send", "recv_reduce"} if operation.kind == "rrc" else {"send", "recv"}
        if (operation.kind not in {"send", "rrc"} or len(members) != 2
                or {item.role for item in members} != required or any(item.operation != operation for item in members)):
            raise SemanticError("physical endpoint pair is missing, duplicated or inconsistent")
        operations.append(operation)
    return Executable(tuple(sorted(operations, key=lambda operation: operation.operation_id)))


@dataclass(frozen=True)
class ProgressEvent:
    time_us: float
    kind: str
    operation_id: str


@dataclass(frozen=True)
class ProgressSegment:
    operation_id: str
    start_us: float
    end_us: float
    rate_bytes_per_us: float
    source: str
