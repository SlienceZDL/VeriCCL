from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import math
from types import MappingProxyType
from typing import Mapping

from vericcl.errors import SemanticError


SOURCES = frozenset({"initial", "estimated", "measured", "history"})


def finite(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise SemanticError(name + " must be finite and " + ("positive" if positive else "nonnegative"))
    return float(value)


def measurement_time(source, stamp):
    if source in {"measured", "history"} and stamp is None:
        raise SemanticError("measured costs require their original measurement time")
    if stamp is not None:
        try:
            parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError, TypeError) as error:
            raise SemanticError("invalid measurement timestamp") from error
        if parsed.tzinfo is None:
            raise SemanticError("measurement timestamp needs a timezone")


@dataclass(frozen=True)
class ServiceCost:
    alpha_us: float
    beta_us_per_byte: float
    source: str = "initial"
    min_bytes: int = 1
    max_bytes: int | None = None
    measured_at: str | None = None
    qualified: bool = True

    def __post_init__(self):
        object.__setattr__(self, "alpha_us", finite(self.alpha_us, "alpha_us"))
        object.__setattr__(self, "beta_us_per_byte", finite(self.beta_us_per_byte, "beta_us_per_byte", positive=True))
        if self.source not in SOURCES or (self.source in {"measured", "history"} and not self.qualified):
            raise SemanticError("invalid or unqualified cost source")
        if isinstance(self.min_bytes, bool) or not isinstance(self.min_bytes, int) or self.min_bytes < 1:
            raise SemanticError("cost size domain must start at a positive byte count")
        if self.max_bytes is not None and (isinstance(self.max_bytes, bool) or not isinstance(self.max_bytes, int) or self.max_bytes < self.min_bytes):
            raise SemanticError("cost size domain is invalid")
        measurement_time(self.source, self.measured_at)

    def covers(self, size: int) -> bool:
        return self.min_bytes <= size and (self.max_bytes is None or size <= self.max_bytes)


@dataclass(frozen=True)
class CostModel:
    version: str
    services: Mapping[str, tuple[ServiceCost, ...]]
    resources: Mapping[str, float] = field(default_factory=dict)
    profiles: tuple = ()
    context: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        from vericcl.performance.profiles import RateProfile
        if not isinstance(self.version, str) or not self.version:
            raise SemanticError("cost model needs a version identity")
        services = {}
        for key, values in self.services.items():
            if not isinstance(key, str) or not key:
                raise SemanticError("cost component key must be nonempty")
            values = tuple(values)
            if not values or not all(isinstance(value, ServiceCost) for value in values):
                raise SemanticError("service cost component must contain ServiceCost records")
            values = tuple(sorted(values, key=lambda value: value.min_bytes))
            for left, right in zip(values, values[1:]):
                if left.max_bytes is None or left.max_bytes >= right.min_bytes:
                    raise SemanticError("service size domains overlap")
            services[key] = values
        resources = {key: finite(value, "resource capacity", positive=True) for key, value in self.resources.items()}
        if any(not isinstance(key, str) or not key for key in resources):
            raise SemanticError("resource identity must be nonempty")
        profiles = tuple(self.profiles)
        if not all(isinstance(profile, RateProfile) for profile in profiles):
            raise SemanticError("model profiles must be RateProfile records")
        if len({p.profile_id for p in profiles}) != len(profiles):
            raise SemanticError("duplicate profile identity")
        object.__setattr__(self, "services", MappingProxyType(services))
        object.__setattr__(self, "resources", MappingProxyType(resources))
        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(self, "context", MappingProxyType(dict(self.context)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def service_for(self, key, size):
        return next((value for value in self.services.get(key, ()) if value.covers(size)), None)

    def with_updates(self, version, *, services=None, resources=None, profiles=None):
        if version == self.version:
            raise SemanticError("a model update needs a new version identity")
        updated_services, updated_resources = dict(self.services), dict(self.resources)
        changes = []
        for key, value in (services or {}).items():
            value = tuple(value)
            old = self.services.get(key)
            updated_services[key] = tuple(value)
            changes.append({"component": "service:" + key,
                            "before": None if old is None else [asdict(c) for c in old],
                            "after": [asdict(c) for c in value]})
        for key, value in (resources or {}).items():
            changes.append({"component": "resource:" + key, "before": self.resources.get(key), "after": value})
            updated_resources[key] = value
        updated_profiles = self.profiles if profiles is None else tuple(profiles)
        if profiles is not None:
            changes.append({"component": "profiles", "before": [p.as_json() for p in self.profiles],
                            "after": [p.as_json() for p in updated_profiles]})
        return CostModel(version, updated_services, updated_resources, updated_profiles, self.context, self.metadata), tuple(changes)
