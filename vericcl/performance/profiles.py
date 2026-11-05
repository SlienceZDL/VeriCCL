from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from vericcl.errors import SemanticError


@dataclass(frozen=True)
class RateProfile:
    profile_id: str
    mode: tuple[str, ...]
    rates: Mapping[str, float]
    resources: tuple[str, ...] = ()
    min_bytes: int = 1
    max_bytes: int | None = None
    source: str = "measured"
    measured_at: str | None = None
    startup_us: Mapping[str, float] = field(default_factory=dict)
    flow_resources: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self):
        from vericcl.performance.model import SOURCES, finite, measurement_time
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise SemanticError("profile needs a stable identity")
        mode = tuple(sorted(self.mode))
        if not mode or any(not isinstance(flow, str) or not flow for flow in mode) or len(set(mode)) != len(mode):
            raise SemanticError("profile mode needs unique semantic flow identities")
        if set(self.rates) != set(mode):
            raise SemanticError("profile rate vector must cover the exact concurrent mode")
        if self.source not in SOURCES:
            raise SemanticError("invalid profile source")
        measurement_time(self.source, self.measured_at)
        if (isinstance(self.min_bytes, bool) or not isinstance(self.min_bytes, int) or self.min_bytes < 1
                or (self.max_bytes is not None and (isinstance(self.max_bytes, bool)
                    or not isinstance(self.max_bytes, int) or self.max_bytes < self.min_bytes))):
            raise SemanticError("invalid profile payload domain")
        if any(not isinstance(resource, str) or not resource for resource in self.resources):
            raise SemanticError("invalid profile resource identity")
        if set(self.startup_us) - set(mode):
            raise SemanticError("startup vector uses an unknown flow")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "rates", MappingProxyType({k: finite(v, "measured rate", positive=True) for k, v in self.rates.items()}))
        object.__setattr__(self, "startup_us", MappingProxyType({k: finite(v, "startup") for k, v in self.startup_us.items()}))
        object.__setattr__(self, "resources", tuple(sorted(set(self.resources))))
        membership = {key: tuple(sorted(set(values))) for key, values in self.flow_resources.items()}
        if membership and (set(membership) != set(mode) or
                           {resource for values in membership.values() for resource in values} != set(self.resources)):
            raise SemanticError('profile resource membership must cover its exact mode and resource union')
        object.__setattr__(self, 'flow_resources', MappingProxyType(membership))

    def as_json(self):
        return {"profile_id": self.profile_id, "mode": list(self.mode), "rates": dict(self.rates),
                "resources": list(self.resources), "min_bytes": self.min_bytes, "max_bytes": self.max_bytes,
                "source": self.source, "measured_at": self.measured_at, "startup_us": dict(self.startup_us),
                'flow_resources': {key: list(values) for key, values in self.flow_resources.items()}}

    def applies(self, operations) -> bool:
        operations = tuple(operations)
        mode = tuple(sorted(operation.flow_id for operation in operations))
        resources = tuple(sorted({resource for operation in operations for resource in operation.resources}))
        membership_matches = not self.flow_resources or all(tuple(sorted(op.resources)) == self.flow_resources.get(op.flow_id) for op in operations)
        return (mode == self.mode and resources == self.resources and membership_matches
                and all(self.min_bytes <= op.payload_bytes and (self.max_bytes is None or op.payload_bytes <= self.max_bytes)
                        for op in operations))
