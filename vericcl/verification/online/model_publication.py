"""Component-wise model replacement with explicit applicability identities."""
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
import functools
import math

from vericcl.performance.model import CostModel
from vericcl.performance.profiles import RateProfile
from vericcl.verification.search_coverage import digest


@dataclass(frozen=True)
class ComponentUpdates:
    services: dict = field(default_factory=dict)
    profiles: tuple = ()
    signatures: dict = field(default_factory=dict)
    issues: tuple = ()
    records: tuple = ()


def resource_envelopes(profiles, baseline):
    """Shared-resource capacities implied by qualified measured profiles.

    The aggregate rate of the flows a measured mode puts on a resource is a
    lower bound on its capacity; it equals the capacity only when the
    resource saturated. A resource counts as saturated when its largest
    measured concurrency did not raise the aggregate by more than 10% over
    the next smaller one. Otherwise the input capacity is kept unless a
    measurement already exceeded it. (Using every envelope as the capacity
    let a single-flow measurement - bounded by its own connection - define a
    whole host's shared limit.)
    """
    by_count = {}
    for profile in profiles:
        if profile.source not in {'measured', 'history'} or not profile.flow_resources:
            continue
        for resource in profile.resources:
            members = [flow for flow in profile.mode if resource in profile.flow_resources[flow]]
            rate = sum((Fraction(str(profile.rates[flow])) for flow in members), Fraction(0))
            counts = by_count.setdefault(resource, {})
            counts[len(members)] = max(counts.get(len(members), Fraction(0)), rate)
    envelopes = {}
    for resource, counts in by_count.items():
        largest = max(counts)
        smaller = [count for count in counts if count < largest]
        observed = max(counts.values())
        saturated = bool(smaller) and largest > 1 and counts[largest] <= Fraction(11, 10) * counts[max(smaller)]
        if saturated or resource not in baseline:
            envelopes[resource] = observed
        else:
            envelopes[resource] = max(observed, Fraction(str(baseline[resource])))
    return envelopes


def profile_key(profile):
    return _profile_key(tuple(profile.mode), tuple(profile.resources))


@functools.lru_cache(maxsize=65536)
def _profile_key(mode, resources):
    return 'profile:' + digest({'mode': list(mode), 'resources': list(resources)})


def _by_key(profiles):
    grouped = {}
    for profile in profiles:
        grouped.setdefault(profile_key(profile), []).append(profile)
    return grouped


def _overlay(levels):
    entries = [piece for level in levels for piece in level]
    if not entries:
        return ()
    boundaries = sorted({x for piece in entries for x in
                         (piece.min_bytes, None if piece.max_bytes is None else piece.max_bytes + 1) if x is not None})
    selected = []
    for index, begin in enumerate(boundaries):
        end = boundaries[index + 1] - 1 if index + 1 < len(boundaries) else None
        winner = next((piece for level in levels for piece in level if piece.min_bytes <= begin and
                       (piece.max_bytes is None or begin <= piece.max_bytes)), None)
        if winner is None:
            continue
        if winner.max_bytes is not None:
            end = min(end, winner.max_bytes) if end is not None else winner.max_bytes
        value = replace(winner, min_bytes=begin, max_bytes=end)
        # Splitting a rate profile needs a unique stable profile id.
        if isinstance(value, RateProfile):
            value = replace(value, profile_id='rate-' + digest({**value.as_json(), 'profile_id': None})[:24])
        if selected and selected[-1].max_bytes == begin - 1:
            left = asdict(selected[-1]) if not isinstance(value, RateProfile) else selected[-1].as_json()
            right = asdict(value) if not isinstance(value, RateProfile) else value.as_json()
            for item in (left, right):
                item.pop('min_bytes'); item.pop('max_bytes'); item.pop('profile_id', None)
            if left == right:
                selected[-1] = replace(selected[-1], max_bytes=end)
                continue
        selected.append(value)
    return tuple(replace(value, profile_id='rate-' + digest({**value.as_json(), 'profile_id': None})[:24])
                 if isinstance(value, RateProfile) else value for value in selected)


def model_json(model):
    return {'version': model.version, 'services': {k: [asdict(v) for v in values] for k, values in model.services.items()},
            'resources': dict(model.resources), 'profiles': [p.as_json() for p in model.profiles],
            'context': dict(model.context), 'metadata': dict(model.metadata)}


_SERIALIZED = {}


def _cached_json(model):
    """model_json of a published model object, computed once.

    Sequential publications compare each new model with the previous one; the
    previous one is the object returned by the last call. Keyed by identity
    (the object is retained with its value), never by version string.
    """
    entry = _SERIALIZED.get(id(model))
    if entry is not None and entry[0] is model:
        return entry[1]
    value = model_json(model)
    while len(_SERIALIZED) >= 4:
        _SERIALIZED.pop(next(iter(_SERIALIZED)))
    _SERIALIZED[id(model)] = (model, value)
    return value


def model_changes(before, after):
    """Changed components only: per service key, per profile id, per resource."""
    changes = []
    for category in ('services', 'profiles', 'resources', 'context', 'metadata'):
        old, new = before[category], after[category]
        if old == new:
            continue
        if category == 'profiles':
            old = {item['profile_id']: item for item in old}
            new = {item['profile_id']: item for item in new}
        if isinstance(old, dict) and isinstance(new, dict):
            changed = {str(key): {'before': old.get(key), 'after': new.get(key)}
                       for key in sorted(set(old) | set(new), key=str) if old.get(key) != new.get(key)}
            changes.append({'component': category, 'changed': changed})
        else:
            changes.append({'component': category, 'before': old, 'after': new})
    return changes


def persist_model(run_context, model, parent_json, *, previous_version, issues=(), records=()):
    """Write a model version and its component-level difference from a parent."""
    after = _cached_json(model)
    run_context.write_json('models/' + model.version + '.json', after)
    run_context.write_json('models/' + model.version + '-diff.json', {
        'previous_version': previous_version, 'version': model.version,
        'changes': model_changes(parent_json, after), 'issues': list(issues), 'qualification_records': list(records)})


def publish_model(old, updates, *, initial=None, history=None, current_signatures=None, run_context=None):
    """New qualified -> applicable current -> explicit history -> initial.

    Callers supply current identities after a runtime/topology change. Unchanged
    task-local identities never expire merely because wall time has passed.
    """
    if not isinstance(updates, ComponentUpdates):
        raise ValueError('model publication requires ComponentUpdates')
    historical = history or ComponentUpdates()
    stale = {key for key, signature in updates.signatures.items()
             if current_signatures is not None and key in current_signatures and current_signatures[key] != signature}
    if stale:
        updates = replace(updates,
            services={key: values for key, values in updates.services.items() if 'service:' + key not in stale},
            profiles=tuple(p for p in updates.profiles if profile_key(p) not in stale),
            signatures={key: value for key, value in updates.signatures.items() if key not in stale},
            issues=updates.issues + tuple({'component': key, 'reason': 'measurement_applicability_changed'} for key in sorted(stale)))
    previous_signatures = dict(old.metadata.get('calibration_signatures', {}))
    signatures = dict(previous_signatures)
    signatures.update(current_signatures or {})
    signatures.update(updates.signatures)
    for key, values in updates.services.items():
        if not values or any(v.source != 'measured' or not v.qualified for v in values) or 'service:' + key not in updates.signatures:
            raise ValueError('new service measurements need qualified values and an applicability identity')
    for value in updates.profiles:
        if value.source != 'measured' or profile_key(value) not in updates.signatures:
            raise ValueError('new rate profiles need a measured source and applicability identity')
    services = {}
    keys = set(old.services) | set(updates.services) | set(historical.services) | set(initial.services if initial else ())
    for key in sorted(keys):
        component = 'service:' + key
        applicable = component in previous_signatures and previous_signatures.get(component) == signatures.get(component)
        current = tuple(v for v in old.services.get(key, ()) if v.source in {'measured', 'history'}) if applicable else ()
        cached = historical.services.get(key, ()) if historical.signatures.get(component) == signatures.get(component) else ()
        baseline = tuple(v for v in (initial or old).services.get(key, ()) if v.source in {'initial', 'estimated'})
        values = _overlay((updates.services.get(key, ()), current, cached, baseline))
        if values:
            services[key] = values
    profiles = []
    # Profiles are grouped by component key once; filtering every profile list
    # per key was quadratic in the model size (seconds per publication).
    new_by, old_by, cached_by = _by_key(updates.profiles), _by_key(old.profiles), _by_key(historical.profiles)
    base_by = old_by if initial is None or initial is old else _by_key(initial.profiles)
    for key in sorted(set(new_by) | set(old_by) | set(cached_by) | set(base_by)):
        applicable = key in previous_signatures and previous_signatures.get(key) == signatures.get(key)
        new = tuple(new_by.get(key, ()))
        current = tuple(p for p in old_by.get(key, ()) if p.source in {'measured', 'history'}) if applicable else ()
        cached = tuple(cached_by.get(key, ())) if historical.signatures.get(key) == signatures.get(key) else ()
        baseline = tuple(p for p in base_by.get(key, ()) if p.source in {'initial', 'estimated'})
        profiles.extend(_overlay((new, current, cached, baseline)))
    metadata = dict(old.metadata)
    if signatures:
        metadata['calibration_signatures'] = signatures
    baseline_resources = dict((initial or old).resources)
    resources = dict(baseline_resources)
    envelopes = resource_envelopes(profiles, baseline_resources)
    for resource, rate in envelopes.items():
        capacity = float(rate)
        if Fraction(str(capacity)) < rate:
            capacity = math.nextafter(capacity, math.inf)
        resources[resource] = capacity
    if envelopes:
        metadata['resource_capacity_rule'] = 'saturated_envelope_or_lower_bound_of_qualified_profiles'
    else:
        metadata.pop('resource_capacity_rule', None)
    candidate = CostModel(old.version, services, resources, tuple(profiles), old.context, metadata)
    before, after = _cached_json(old), model_json(candidate)
    if before == after:
        return old
    candidate = replace(candidate, version='cost-' + digest({'parent': old.version, 'contents': after})[:24])
    after['version'] = candidate.version
    _SERIALIZED[id(candidate)] = (candidate, after)
    if run_context is not None:
        persist_model(run_context, candidate, before, previous_version=old.version,
                      issues=updates.issues, records=updates.records)
    return candidate
