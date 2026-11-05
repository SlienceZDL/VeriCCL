"""Advance remaining bytes across resource-mode changes without restarting work."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from functools import lru_cache
import heapq
from types import MappingProxyType

from gmpy2 import mpq

from vericcl.errors import SemanticError
from vericcl.performance.events import Executable, ProgressEvent, ProgressSegment
from vericcl.performance.model import CostModel


@dataclass(frozen=True)
class Prediction:
    model_version: str
    t_pred_us: float | None
    output_ready_us: float | None
    start_times: dict
    end_times: dict
    events: tuple[ProgressEvent, ...]
    segments: tuple[ProgressSegment, ...]
    used_components: frozenset[str]
    missing_components: frozenset[str]
    unavailable_operations: tuple[str, ...]
    model_coverage: str
    exact_completion_ratio: tuple[int, int] | None = None
    unmeasured_modes: tuple[tuple[str, ...], ...] = ()


def _exact_number(value):
    # Preserve the serialized decimal contract instead of converting a binary
    # float directly to its exact (but different) rational representation.
    return mpq(value) if isinstance(value, int) else mpq(str(value))


def _exact_float(value):
    # Python integer division matches Fraction.__float__ and is independent of
    # the caller's GMP floating-point precision and rounding context.
    return int(value.numerator) / int(value.denominator)


def _acyclic(operations, lease=None):
    degrees = {op.operation_id: len(op.dependencies) for op in operations}
    successors = {key: [] for key in degrees}
    for operation in operations:
        for predecessor in operation.dependencies:
            successors[predecessor].append(operation.operation_id)
    remaining = dict(degrees)
    ready = deque(key for key, degree in remaining.items() if degree == 0)
    completed = 0
    while ready:
        if lease is not None:
            lease.check()
        key = ready.popleft()
        completed += 1
        for follower in successors[key]:
            remaining[follower] -= 1
            if remaining[follower] == 0:
                ready.append(follower)
    if completed != len(degrees):
        raise SemanticError("executable dependency cycle")
    return degrees, successors


def _components(operations, lease=None):
    remaining = list(operations)
    result = []
    while remaining:
        if lease is not None:
            lease.check()
        group = [remaining.pop(0)]
        resources = set(group[0].resources)
        changed = True
        while changed:
            changed = False
            for operation in tuple(remaining):
                if resources.intersection(operation.resources):
                    group.append(operation)
                    resources.update(operation.resources)
                    remaining.remove(operation)
                    changed = True
        result.append(group)
    return result


def _profile(group, model):
    matches = [profile for profile in model.profiles if profile.applies(group)]
    if len(matches) > 1:
        raise SemanticError("multiple profiles apply to the same concurrency mode")
    return matches[0] if matches else None


def _rates(active, costs, model, used, number, unmeasured, lease=None):
    rates, sources = {}, {}
    communication = []
    for operation in active:
        if operation.kind in {"send", "rrc"}:
            communication.append(operation)
        else:
            rates[operation.operation_id] = 1 / number(costs[operation.operation_id].beta_us_per_byte)
            sources[operation.operation_id] = costs[operation.operation_id].source
    for group in _components(communication, lease):
        if len({operation.flow_id for operation in group}) != len(group):
            raise SemanticError("concurrent operations share a flow identity; serialize their TB work")
        profile = _profile(group, model)
        resources = {resource for operation in group for resource in operation.resources}
        for resource in resources:
            used.add("resource:" + resource)
        if profile is not None:
            for resource in resources:
                assigned = sum(number(profile.rates[op.flow_id]) for op in group if resource in op.resources)
                if assigned > number(model.resources[resource]):
                    raise SemanticError("measured rate vector conflicts with resource capacity " + resource)
            used.add("profile:" + profile.profile_id)
            for operation in group:
                rates[operation.operation_id] = number(profile.rates[operation.flow_id])
                sources[operation.operation_id] = profile.source
        else:
            if len(group) > 1:
                unmeasured.add(tuple(sorted(op.flow_id for op in group)))
            for operation in group:
                limits = [1 / number(costs[operation.operation_id].beta_us_per_byte)]
                for resource in operation.resources:
                    count = sum(resource in member.resources for member in group)
                    limits.append(number(model.resources[resource]) / count)
                rates[operation.operation_id] = min(limits)
                sources[operation.operation_id] = ("initial_sharing_estimate" if len(group) > 1 or min(limits) < limits[0]
                                                   else costs[operation.operation_id].source)
    return rates, sources


def predict(executable: Executable, model: CostModel, *, exact=False, lease=None) -> Prediction:
    """Replay fixed decisions; exact mode treats serialized decimals as rationals.

    Exact mode removes floating arithmetic from event ordering and completion
    comparison. It certifies only this model evaluation, not search coverage or
    agreement between the model and measured hardware behavior.
    """
    if lease is not None:
        lease.check()
    number = lru_cache(maxsize=None)(_exact_number) if exact else float
    as_float = _exact_float if exact else float
    if dict(executable.context) != dict(model.context):
        raise SemanticError("cost model applicability context differs from the executable")
    operations = {operation.operation_id: operation for operation in executable.operations}
    degrees, successors = _acyclic(executable.operations, lease)
    costs, unavailable, missing, used = {}, set(), set(), set()
    for key, operation in operations.items():
        if operation.kind == "nop":
            continue
        cost = model.service_for(operation.cost_key, operation.payload_bytes)
        if cost is None:
            unavailable.add(key)
            missing.add("service:" + operation.cost_key)
        else:
            costs[key] = replace(cost, source=operation.source_override) if operation.source_override else cost
        for resource in operation.resources:
            if resource not in model.resources:
                unavailable.add(key)
                missing.add("resource:" + resource)
    changed = True
    while changed:
        if lease is not None:
            lease.check()
        changed = False
        affected_resources = {resource for key in unavailable for resource in operations[key].resources}
        for key, operation in operations.items():
            if key not in unavailable and (set(operation.dependencies) & unavailable or set(operation.resources) & affected_resources):
                unavailable.add(key)
                changed = True
    pending = set(operations) - unavailable
    releases = [(number(operations[key].not_before_us), key)
                for key in pending if degrees[key] == 0]
    heapq.heapify(releases)

    def complete(key, time):
        ends[key] = time
        events.append(ProgressEvent(time, "complete", key))
        for index, follower in enumerate(successors[key]):
            if lease is not None and index % 128 == 0:
                lease.check()
            if follower not in pending:
                continue
            degrees[follower] -= 1
            if degrees[follower] == 0:
                heapq.heappush(releases, (number(operations[follower].not_before_us), follower))

    startup, active, remaining = {}, set(), {}
    starts, ends, events, segments = {}, {}, [], []
    sources_used, unmeasured = set(), set()
    now = number(0)
    while pending or startup or active:
        if lease is not None:
            lease.check()
        # Finish startup before launching newly dependent operations at this
        # event. Alpha remains associated with the operation's original payload.
        for key in sorted(tuple(startup)):
            if startup[key] <= now:
                del startup[key]
                active.add(key)
                events.append(ProgressEvent(now, "startup_complete", key))
        ready = []
        while releases and releases[0][0] <= now:
            _, key = heapq.heappop(releases)
            ready.append(key)
        ready.sort()
        # Select a startup coefficient once, using the launch-time mode. A
        # competitor arriving later changes rates, never repays this startup.
        launch_profiles = {}
        present = active | startup.keys() | set(ready)
        for group in _components([operations[key] for key in sorted(present)
                                  if operations[key].kind in {"send", "rrc"}]):
            profile = _profile(group, model)
            if profile is not None:
                for operation in group:
                    if operation.flow_id in profile.startup_us:
                        launch_profiles[operation.operation_id] = profile
        for key in ready:
            operation = operations[key]
            pending.remove(key)
            starts[key] = now
            events.append(ProgressEvent(now, "start", key))
            if operation.kind == "nop":
                complete(key, now)
                continue
            cost = costs[key]
            used.add("service:" + operation.cost_key)
            profile = launch_profiles.get(key)
            if profile is None:
                alpha, source = cost.alpha_us, cost.source
            else:
                alpha, source = profile.startup_us[operation.flow_id], profile.source
                used.add("profile:" + profile.profile_id)
            sources_used.add(source)
            remaining[key] = number(operation.payload_bytes)
            startup[key] = now + number(alpha)
        # New zero-startup work and NOP chains are resolved at this same time.
        if any(end <= now for end in startup.values()) or (releases and releases[0][0] <= now):
            continue
        rates, rate_sources = _rates([operations[key] for key in sorted(active)], costs, model, used, number, unmeasured, lease)
        sources_used.update(rate_sources.values())
        finish_times = {key: now + remaining[key] / rates[key] for key in active}
        future = list(finish_times.values()) + list(startup.values())
        if releases:
            future.append(releases[0][0])
        if not future:
            if pending:
                raise SemanticError("executable is stalled by unresolved work")
            break
        next_time = min(future)
        if next_time <= now:
            raise SemanticError("model time precision cannot represent positive progress")
        delta = next_time - now
        for key in sorted(active):
            segments.append(ProgressSegment(key, as_float(now), as_float(next_time), as_float(rates[key]), rate_sources[key]))
            remaining[key] = max(number(0), remaining[key] - rates[key] * delta)
        now = next_time
        for key in sorted(tuple(active)):
            if finish_times[key] == next_time:
                active.remove(key)
                complete(key, now)
    t_pred = None if unavailable else max(ends.values(), default=number(0))
    output_ready = (None if set(executable.output_operations) & unavailable
                    else max((ends[key] for key in executable.output_operations), default=number(0)))
    coverage = ("unavailable" if unavailable else "not_required" if not sources_used
                else "measured" if sources_used <= {"measured", "history"}
                else "partial" if sources_used & {"measured", "history"} else "initial")
    ratio = (int(t_pred.numerator), int(t_pred.denominator)) if exact and t_pred is not None else None
    return Prediction(model.version, None if t_pred is None else as_float(t_pred),
                      None if output_ready is None else as_float(output_ready),
                      MappingProxyType({key: as_float(value) for key, value in starts.items()}),
                      MappingProxyType({key: as_float(value) for key, value in ends.items()}),
                      tuple(ProgressEvent(as_float(event.time_us), event.kind, event.operation_id) for event in events),
                      tuple(segments), frozenset(used), frozenset(missing), tuple(sorted(unavailable)), coverage,
                      ratio, tuple(sorted(unmeasured)))
