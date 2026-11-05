"""Scoped drift estimates; one size never identifies alpha and beta separately."""
from dataclasses import replace
from datetime import datetime, timezone
import math

from vericcl.performance.model import CostModel, ServiceCost
from vericcl.verification.online.model_publication import _overlay, model_json
from vericcl.verification.search_coverage import digest


def update_live_model(model, events, *, payload_bytes):
    services = dict(model.services); resources = dict(model.resources); factors = {}; external = {}
    for event in events:
        for key, change in event.changes.items():
            if key.startswith('service:') and key[8:] in services:
                factor = 1 + change
                if not math.isfinite(factor) or factor <= 0:
                    raise ValueError('live drift must retain a positive service cost')
                factors[key[8:]] = factor
        if event.kind == 'model_update':
            external.update(event.evidence['metrics'])
    if not factors and not external:
        return model
    stamp = datetime.now(timezone.utc).isoformat()
    for key, values in external.items():
        old = model.service_for(key, payload_bytes)
        if old is not None and values['min_bytes'] <= payload_bytes <= values['max_bytes']:
            factors[key] = (values['alpha_us'] + values['beta_us_per_byte'] * payload_bytes) / (old.alpha_us + old.beta_us_per_byte * payload_bytes)
    for key, factor in factors.items():
        old = model.service_for(key, payload_bytes)
        if old is None:
            continue
        point = replace(old, alpha_us=old.alpha_us * factor, beta_us_per_byte=old.beta_us_per_byte * factor,
                        min_bytes=payload_bytes, max_bytes=payload_bytes, source='estimated', measured_at=stamp)
        services[key] = _overlay(((point,), services[key]))
    for key, values in external.items():
        point = ServiceCost(**values, source='estimated', measured_at=stamp)
        services[key] = _overlay(((point,), services.get(key, ())))
    profiles = []
    for profile in model.profiles:
        if not profile.min_bytes <= payload_bytes or (profile.max_bytes is not None and profile.max_bytes < payload_bytes):
            profiles.append(profile); continue
        rates, startup = dict(profile.rates), dict(profile.startup_us); changed = False
        for flow in profile.mode:
            parts = flow.split(':')
            key = ':'.join((['link'] if parts[0] == 'send' else [parts[0]]) + parts[1:3])
            if key in factors:
                rates[flow] /= factors[key]
                if flow in startup: startup[flow] *= factors[key]
                changed = True
        if changed:
            point = replace(profile, rates=rates, startup_us=startup, source='estimated', measured_at=stamp,
                            min_bytes=payload_bytes, max_bytes=payload_bytes)
            profiles.extend(_overlay(((point,), (profile,))))
        else:
            profiles.append(profile)
    # Sharing updates retain only already known mode shapes and apply at S.
    # Resource envelopes are retained because they have no size-domain field.
    metadata = dict(model.metadata)
    metadata['live_update'] = {'scope': 'observed_payload_only_relative_drift_estimate',
        'payload_bytes': payload_bytes, 'factors': factors, 'external_costs': external,
        'observed_at': stamp, 'parent_model': model.version,
        'assumption': 'software duration ratios scale the previous alpha/beta shape; no independent parameter fit',
        'event_sources': [event.evidence.get('source', 'software_device_counters') for event in events],
        'unobserved_components': 'retained; no zero filling and no new concurrency modes'}
    after = CostModel(model.version, services, resources, tuple(profiles), model.context, metadata)
    return replace(after, version='live-' + digest(model_json(after))[:24])
