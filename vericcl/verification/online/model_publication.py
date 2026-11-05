"""Component-wise model replacement with explicit applicability identities."""
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
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


def profile_key(profile):
    return 'profile:' + digest({'mode': list(profile.mode), 'resources': list(profile.resources)})


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
    all_profiles = (*old.profiles, *updates.profiles, *historical.profiles, *(initial.profiles if initial else ()))
    for key in sorted({profile_key(p) for p in all_profiles}):
        applicable = key in previous_signatures and previous_signatures.get(key) == signatures.get(key)
        new = tuple(p for p in updates.profiles if profile_key(p) == key)
        current = tuple(p for p in old.profiles if profile_key(p) == key and p.source in {'measured', 'history'}) if applicable else ()
        cached = tuple(p for p in historical.profiles if profile_key(p) == key) if historical.signatures.get(key) == signatures.get(key) else ()
        baseline = tuple(p for p in (initial or old).profiles if profile_key(p) == key and p.source in {'initial', 'estimated'})
        profiles.extend(_overlay((new, current, cached, baseline)))
    metadata = dict(old.metadata)
    if signatures:
        metadata['calibration_signatures'] = signatures
    resources = dict((initial or old).resources)
    envelopes = {}
    for profile in profiles:
        if profile.source not in {'measured', 'history'} or not profile.flow_resources:
            continue
        for resource in profile.resources:
            rate = sum((Fraction(str(profile.rates[flow])) for flow in profile.mode
                        if resource in profile.flow_resources[flow]), Fraction(0))
            envelopes[resource] = max(envelopes.get(resource, Fraction(0)), rate)
    for resource, rate in envelopes.items():
        capacity = float(rate)
        if Fraction(str(capacity)) < rate:
            capacity = math.nextafter(capacity, math.inf)
        resources[resource] = capacity
    if envelopes:
        metadata['resource_capacity_rule'] = 'envelope_of_qualified_applicable_profile_vectors'
    else:
        metadata.pop('resource_capacity_rule', None)
    candidate = CostModel(old.version, services, resources, tuple(profiles), old.context, metadata)
    before, after = model_json(old), model_json(candidate)
    if before == after:
        return old
    candidate = replace(candidate, version='cost-' + digest({'parent': old.version, 'contents': after})[:24])
    if run_context is not None:
        run_context.write_json('models/' + candidate.version + '.json', model_json(candidate))
        changes = []
        for category in ('services', 'profiles', 'resources', 'context', 'metadata'):
            if before[category] != after[category]:
                changes.append({'component': category, 'before': before[category], 'after': after[category]})
        run_context.write_json('models/' + candidate.version + '-diff.json', {
            'previous_version': old.version, 'version': candidate.version, 'changes': changes,
            'issues': updates.issues, 'qualification_records': updates.records})
    return candidate
