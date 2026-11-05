"""Incremental calibration: independent failure domains and budgeted fit follow-ups."""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from vericcl.execution.budget import BudgetExhausted
from vericcl.execution.events import emit_event
from vericcl.verification.search_coverage import digest
from .component_cache import CalibrationQuery, lookup_calibration, save_calibration
from .component_fitting import FitPolicy, fit_components, _identity
from .model_publication import ComponentUpdates, publish_model, profile_key
from .probe_jobs import plan_probes, execute_probe, _mode


@dataclass(frozen=True)
class CampaignResult:
    model: object
    observations: tuple
    confirmations: tuple
    issues: tuple
    status: str


def _cache_path(context):
    setting = context.inputs.request_config['sketch']['history']
    if not setting['calibration_reuse']:
        return None
    path = Path(setting['calibration_index']).expanduser()
    if not path.is_absolute():
        # Relative shared caches live in the case, never the caller's CWD.
        path = context.run_context.root.parent.parent / path
    return path.resolve()


def run_calibration(context, model, *, initial=None, driver=None, jobs=None):
    policy = FitPolicy(context.inputs.request_config['sketch']['calibration']['max_relative_fit_error'], lease=context.lease)
    original, current = initial or model, model
    observations, confirmations, issues = [], [], []
    cache_path = _cache_path(context)
    settings = context.inputs.request_config['sketch']['history']
    status = 'completed'

    def resource_state():
        results = [result for _, result in (*observations, *confirmations)]
        active = sorted(context.run_context.active_jobs) if context.run_context is not None else []
        clean = all(result.cleanup_confirmed is True for result in results)
        released = all(result.gpu_released is True for result in results)
        return dict(cleanup_confirmed=clean, gpu_released=released, active_jobs=active,
                    safe_to_continue=clean and released and not active, observations=len(results))

    def finish():
        state = resource_state()
        if context.run_context is not None:
            context.run_context.write_json('reports/calibration-resource-state.json',
                dict(campaign_status=status, **state), replace=True)
        return CampaignResult(current, tuple(observations), tuple(confirmations), tuple(issues), status)

    def unresolved(job=None):
        nonlocal status
        state = resource_state()
        if state['safe_to_continue']:
            return False
        status = 'cleanup_unconfirmed'
        issue = dict(reason=status, job_id=None if job is None else job.job_id, **state)
        issues.append(issue)
        if context.run_context is not None:
            emit_event(context.run_context, 'calibration_cleanup_unconfirmed', issue)
        return True

    try:
        if unresolved():
            return finish()
        jobs = tuple(plan_probes(context) if jobs is None else jobs)
        groups = {}
        for job in jobs:
            key = (job.category, job.host_scope, job.flows, job.resources)
            groups.setdefault(key, []).append(job)
        # Keep independent intra-host links ahead of local work, inter-host and
        # shared tests. Stable ties retain the planner's channel-count order;
        # opaque job hashes must not reorder the physical coverage priorities.
        ordered = sorted(groups.values(), key=lambda group: (
            group[0].category == 'shared', group[0].host_scope == 'inter',
            group[0].category != 'link'))
        for planned in ordered:
            primary, confirmation = [], []
            for job in planned:
                context.lease.check()
                result = execute_probe(job, context, driver)
                primary.append((job, result)); observations.append((job, result))
                if unresolved(job):
                    return finish()
            attempted = {job.payload_bytes for job, _ in primary}
            confirmed = set()
            while True:
                context.lease.check()
                updates = fit_components(primary, policy, confirmations=confirmation)
                history_services, history_profiles, history_signatures = {}, [], {}
                # Current native identities are taken from qualified provenance,
                # including failed-fit groups; approximate schedule reuse never
                # authorizes absolute calibration reuse.
                current_signatures = dict(updates.signatures)
                if cache_path is not None:
                    for record in updates.records:
                        if 'summary' not in record:
                            continue
                        identity = record['identity']
                        job = next(item for item, result in primary if item.job_id == record['job_id'])
                        from .component_fitting import _service_key
                        keys = []
                        if len(job.flows) == 1 and job.flows[0].channel == 0:
                            keys.append('service:' + _service_key(job.flows[0]))
                        if all(flow.kind in {'send', 'rrc'} for flow in job.flows):
                            keys.append('profile:' + digest({'mode': sorted(flow.flow_id for flow in job.flows),
                                                            'resources': sorted(job.resources)}))
                        for key in keys:
                            current_signatures[key] = identity
                            cached = lookup_calibration(CalibrationQuery(cache_path, key, identity,
                                enabled=True, max_age_s=settings['calibration_max_age_s']))
                            if cached.status == 'hit':
                                history_signatures[key] = identity
                                if key.startswith('service:'):
                                    history_services[key[8:]] = cached.values
                                else:
                                    history_profiles.extend(cached.values)
                historical = ComponentUpdates(history_services, tuple(history_profiles), history_signatures)
                current = publish_model(current, updates, initial=original, history=historical,
                    current_signatures=current_signatures, run_context=context.run_context)
                if cache_path is not None:
                    for key, values in updates.services.items():
                        save_calibration(cache_path, 'service:' + key, updates.signatures['service:' + key], values)
                    profiles = {}
                    for profile in updates.profiles:
                        profiles.setdefault(profile_key(profile), []).append(profile)
                    for key, values in profiles.items():
                        save_calibration(cache_path, key, updates.signatures[key], values)
                followups = []
                prototype = planned[0]
                for issue in updates.issues:
                    for size in issue.get('additional_payloads', ()):
                        if size not in attempted:
                            attempted.add(size); followups.append((size, False))
                    for size in issue.get('confirmation_payloads', ()):
                        if size not in confirmed:
                            confirmed.add(size); followups.append((size, True))
                if context.run_context is not None:
                    emit_event(context.run_context, 'calibration_component', {
                        'mode': [flow.flow_id for flow in prototype.flows], 'model_version': current.version,
                        'issues': updates.issues, 'followups': followups})
                if not followups:
                    issues.extend(updates.issues)
                    break
                for size, is_confirmation in followups:
                    context.lease.check()
                    job = _mode(context, prototype.category, prototype.flows, size)
                    pair = (job, execute_probe(job, context, driver))
                    if is_confirmation:
                        confirmation.append(pair); confirmations.append(pair)
                    else:
                        primary.append(pair); observations.append(pair)
                    if unresolved(job):
                        return finish()
    except BudgetExhausted as error:
        status = 'budget_exhausted'
        issues.append({'reason': str(error), 'completed_observations': len(observations)})
        unresolved()
    return finish()
