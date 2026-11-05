from dataclasses import replace

import pytest

from tests.unit.online.test_probe_jobs import probe_context
from vericcl.verification.online.probe_jobs import ProbeResult, plan_probes


def mathematical_native_contract(job, *, identity='fixture-runtime', factor=1):
    # These are mathematical unit fixtures, never saved as hardware evidence.
    values = {flow.flow_id: factor * (2 + .01 * job.payload_bytes) for flow in job.flows}
    samples = tuple({'round': 1, 'launch': index, 'flow_us': values, 'correct': True, 'gpu_released': True}
                    for index in range(1, job.launches + 1))
    return ProbeResult(job.job_id, 'completed', 'native_msccl', samples,
        cleanup_confirmed=True, gpu_released=True, provenance={
            'measurement_id': job.job_id + '-fixture-' + identity, 'measured_at': '2026-11-04T00:00:00Z',
            'runtime_identity': {'fixture': identity}, 'selected_round': 1})


def test_only_qualified_native_components_are_published_and_profile_membership_is_exact(tmp_path):
    from vericcl.verification.online.component_fitting import FitPolicy, fit_components
    context = probe_context(tmp_path, channels=2, ranks=2)
    jobs = [job for job in plan_probes(context) if job.category == 'link' and job.flows[0].src_rank == 0 and job.flows[0].kind == 'send']
    updates = fit_components([(job, mathematical_native_contract(job)) for job in jobs], FitPolicy(.01))
    assert not updates.issues
    assert updates.services['link:0:1'][0].alpha_us == pytest.approx(2)
    assert len(updates.profiles) == 2
    for profile in updates.profiles:
        assert set(profile.flow_resources) == set(profile.mode)
        assert all(profile.rates[flow] == pytest.approx(100) for flow in profile.mode)
    simulation = [(job, replace(mathematical_native_contract(job), driver_kind='test_simulation')) for job in jobs]
    rejected = fit_components(simulation, FitPolicy(.01))
    assert not rejected.services and not rejected.profiles and len(rejected.issues) == len(jobs)


def test_failed_shared_measurement_cannot_erase_qualified_independent_link(tmp_path):
    from vericcl.verification.online.component_fitting import FitPolicy, fit_components
    context = probe_context(tmp_path, channels=2, ranks=2)
    jobs = [job for job in plan_probes(context) if job.category == 'link' and job.flows[0].src_rank == 0 and job.flows[0].kind == 'send']
    observations = []
    for job in jobs:
        result = mathematical_native_contract(job)
        if len(job.flows) == 2:
            result = replace(result, status='failed')
        observations.append((job, result))
    updates = fit_components(observations, FitPolicy(.01))
    assert 'link:0:1' in updates.services
    assert len(updates.profiles) == 1 and len(updates.profiles[0].mode) == 1
    assert len(updates.issues) == 3


def test_measurement_identity_change_prevents_combining_payloads(tmp_path):
    from vericcl.verification.online.component_fitting import FitPolicy, fit_components
    context = probe_context(tmp_path, channels=1, ranks=2)
    jobs = [job for job in plan_probes(context) if job.category == 'copy' and job.rank_ids == (0,)]
    observations = [(job, mathematical_native_contract(job, identity='A' if index < 2 else 'B')) for index, job in enumerate(jobs)]
    updates = fit_components(observations, FitPolicy(.01))
    assert not updates.services
    assert {issue['reason'] for issue in updates.issues} == {'insufficient_sizes', 'multiple_runtime_identities_require_explicit_selection'}


def test_publication_updates_capacity_envelope_and_repredicts_fixed_executable(tmp_path):
    from vericcl.performance.events import Executable, WorkOperation
    from vericcl.performance.model import CostModel, ServiceCost
    from vericcl.performance.simulation import predict
    from vericcl.verification.online.component_fitting import FitPolicy, fit_components
    from vericcl.verification.online.model_publication import publish_model
    context = probe_context(tmp_path, channels=1, ranks=2)
    jobs = [job for job in plan_probes(context) if job.category == 'link' and job.flows[0].src_rank == 0 and job.flows[0].kind == 'send']
    updates = fit_components([(job, mathematical_native_contract(job)) for job in jobs], FitPolicy(.01))
    initial = CostModel('initial', {'link:0:1': (ServiceCost(1, .1),)}, {'link:0:1': 10})
    new = publish_model(initial, updates, initial=initial)
    assert new.resources['link:0:1'] == pytest.approx(100)
    operation = WorkOperation('send', 'send', 'link:0:1', 1024, resources=('link:0:1',), flow_id='send:0:1:c0')
    executable = Executable((operation,), ('send',))
    before, after = predict(executable, initial, exact=True), predict(executable, new, exact=True)
    assert after.t_pred_us == pytest.approx(12.24) and before.t_pred_us == pytest.approx(103.4)
    assert before.model_version != after.model_version
    assert publish_model(new, type(updates)(), initial=initial) is new


def test_piecewise_confirmation_requires_new_independent_measurement_ids(tmp_path):
    from vericcl.verification.online.component_fitting import FitPolicy, fit_components
    context = probe_context(tmp_path, channels=1, ranks=2)
    base = next(job for job in plan_probes(context) if job.category == 'copy' and job.rank_ids == (0,))
    observations = []
    for size in (100, 200, 300, 1000, 2000, 3000):
        job = replace(base, job_id='fixture-size-' + str(size), payload_bytes=size)
        result = mathematical_native_contract(job)
        duration = 1 + .01 * size if size <= 300 else 8 + .04 * size
        result = replace(result, samples=tuple(sample | {'flow_us': {'copy:0': duration}} for sample in result.samples))
        observations.append((job, result))
    rejected = fit_components(observations, FitPolicy(.01), confirmations=observations)
    assert not rejected.services
    confirmation = [(job, replace(result, provenance=result.provenance | {
        'measurement_id': result.provenance['measurement_id'] + '-new-run'})) for job, result in observations]
    accepted = fit_components(observations, FitPolicy(.01), confirmations=confirmation)
    assert len(accepted.services['copy:0']) == 2
