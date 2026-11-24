from dataclasses import replace
import xml.etree.ElementTree as ET

import pytest

from tests.unit.online.test_probe_jobs import probe_context


def test_probe_xml_uses_independent_slots_and_matching_endpoints(tmp_path):
    from vericcl.verification.online.probe_jobs import plan_probes
    from vericcl.verification.online.probe_xml import build_probe_xml
    context = probe_context(tmp_path, channels=3, ranks=2)
    job = replace(next(j for j in plan_probes(context) if len(j.flows) == 3 and j.flows[0].kind == 'rrc'), chain_steps=1)
    artifact = build_probe_xml(job)
    root = ET.fromstring(artifact.xml)
    assert root.attrib['coll'] == 'custom' and root.attrib['nchunksperloop'] == '3'
    for index, flow in enumerate(job.flows):
        ends = [e for e in artifact.endpoints if e['flow_id'] == flow.flow_id]
        assert {e['role'] for e in ends} == {'send', 'rrc'}
        for end in ends:
            tb = root.find("gpu[@id='%d']/tb[@id='%d']" % (end['rank'], end['tb']))
            step = tb.find('step')
            assert step.attrib['cnt'] == '1' and step.attrib['srcoff'] == str(index)
            assert step.attrib['dstoff'] == str(index) and tb.attrib['chan'] == str(flow.channel)
    assert len(artifact.endpoints) == 6


def test_probe_samples_pair_physical_transfer_once_and_require_all_ranks(tmp_path):
    from vericcl.verification.online.probe_jobs import plan_probes
    from vericcl.verification.online.probe_xml import build_probe_xml
    from vericcl.verification.online.remote import combine_rank_samples
    context = probe_context(tmp_path, channels=1, ranks=2)
    job = replace(next(j for j in plan_probes(context) if j.flows[0].kind == 'rrc'), chain_steps=1)
    artifact = build_probe_xml(job)
    results = []
    for rank in range(2):
        results.append(dict(job_id=job.job_id, rank=rank, status='completed', correct=True,
            gpu_released=True, datatype=job.datatype, reduction_op=job.reduction_op,
            payload_bytes=job.payload_bytes, timed_iterations=job.timed_iterations,
            endpoints=[dict(tb=e['tb'], step=0, samples=job.timed_iterations,
                invalid_samples=0, service_ns=(2 + rank) * 1000 * job.timed_iterations,
                dependency_wait_ns=0, first_start_ns=1000, last_end_ns=9000,
                kind=0 if rank == 0 else 4, channel=0)
                for e in artifact.endpoints if e['rank'] == rank]))
    sample = combine_rank_samples(job, artifact, results)
    assert sample['flow_us'] == {job.flows[0].flow_id: 3.0}
    assert sample['correct'] and sample['gpu_released']
    with pytest.raises(ValueError, match='rank'):
        combine_rank_samples(job, artifact, results[:1])
    results[1]['endpoints'][0]['samples'] -= 1
    with pytest.raises(ValueError, match='sample'):
        combine_rank_samples(job, artifact, results)


def test_probe_correctness_or_release_failure_cannot_publish_sample(tmp_path):
    from vericcl.verification.online.probe_jobs import plan_probes
    from vericcl.verification.online.probe_xml import build_probe_xml
    from vericcl.verification.online.remote import combine_rank_samples
    context = probe_context(tmp_path, channels=1, ranks=2)
    job = next(j for j in plan_probes(context) if j.category == 'copy')
    for result in ({'correct': False, 'gpu_released': True}, {'correct': True, 'gpu_released': False}):
        result.update(job_id=job.job_id, rank=0, status='completed')
        with pytest.raises(ValueError):
            combine_rank_samples(job, build_probe_xml(job), [result])


def test_probe_flow_time_covers_the_launch_wall_time_per_iteration(tmp_path):
    from vericcl.verification.online.probe_jobs import plan_probes
    from vericcl.verification.online.probe_xml import build_probe_xml
    from vericcl.verification.online.remote import combine_rank_samples
    context = probe_context(tmp_path, channels=1, ranks=2)
    job = replace(next(j for j in plan_probes(context) if j.flows[0].kind == 'rrc'), chain_steps=1)
    artifact = build_probe_xml(job)
    def results(wall_s):
        return [dict(job_id=job.job_id, rank=rank, status='completed', correct=True,
            gpu_released=True, datatype=job.datatype, reduction_op=job.reduction_op,
            payload_bytes=job.payload_bytes, timed_iterations=job.timed_iterations, timed_wall_s=wall_s,
            endpoints=[dict(tb=e['tb'], step=0, samples=job.timed_iterations,
                invalid_samples=0, service_ns=(2 + rank) * 1000 * job.timed_iterations,
                dependency_wait_ns=0, first_start_ns=1000, last_end_ns=9000,
                kind=0 if rank == 0 else 4, channel=0)
                for e in artifact.endpoints if e['rank'] == rank]) for rank in range(2)]
    flow = job.flows[0].flow_id
    # Intra-host transfers keep their exact GPU step time.
    assert combine_rank_samples(job, artifact, results(12e-6 * job.timed_iterations))['flow_us'][flow] == 3.0
    job = replace(job, rank_map=tuple(dict(item, host='h%d' % index) for index, item in enumerate(job.rank_map)))
    # Hidden network time: 12 us per iteration although the GPU steps see 3 us.
    sample = combine_rank_samples(job, artifact, results(12e-6 * job.timed_iterations))
    assert sample['flow_us'][flow] == pytest.approx(12.0) and sample['service_us'][flow] == 3.0
    assert sample['wall_scale'] == pytest.approx(4.0)
    # A wall time below the step service (clock granularity) never shortens it.
    assert combine_rank_samples(job, artifact, results(1e-6 * job.timed_iterations))['flow_us'][flow] == 3.0


def test_chain_probe_runs_consecutive_slot_pieces_and_reports_per_step_time(tmp_path):
    from vericcl.verification.online.probe_jobs import plan_probes
    from vericcl.verification.online.probe_xml import build_probe_xml
    from vericcl.verification.online.remote import combine_rank_samples
    context = probe_context(tmp_path, channels=1, ranks=2)
    job = replace(next(j for j in plan_probes(context) if j.flows[0].kind == 'rrc'), chain_steps=4)
    assert job.runtime_payload_bytes == 4 * job.payload_bytes
    artifact = build_probe_xml(job)
    root = ET.fromstring(artifact.xml)
    assert root.attrib['nchunksperloop'] == '4'
    for end in artifact.endpoints:
        tb = root.find("gpu[@id='%d']/tb[@id='%d']" % (end['rank'], end['tb']))
        assert [step.attrib['srcoff'] for step in tb.findall('step')] == ['0', '1', '2', '3']
    assert len(artifact.endpoints) == 8
    def results(wall_s):
        return [dict(job_id=job.job_id, rank=rank, status='completed', correct=True,
            gpu_released=True, datatype=job.datatype, reduction_op=job.reduction_op,
            payload_bytes=job.runtime_payload_bytes, timed_iterations=job.timed_iterations, timed_wall_s=wall_s,
            endpoints=[dict(tb=e['tb'], step=e['step'], samples=job.timed_iterations,
                invalid_samples=0, service_ns=(2 + rank) * 1000 * job.timed_iterations,
                dependency_wait_ns=0, first_start_ns=1000, last_end_ns=9000,
                kind=0 if rank == 0 else 4, channel=0)
                for e in artifact.endpoints if e['rank'] == rank]) for rank in range(2)]
    flow = job.flows[0].flow_id
    # The slower endpoint's chain (4 steps of 3 us) is 3 us per step.
    assert combine_rank_samples(job, artifact, results(None))['flow_us'][flow] == pytest.approx(3.0)
    inter = replace(job, rank_map=tuple(dict(item, host='h%d' % index) for index, item in enumerate(job.rank_map)))
    # 40 us of wall time per iteration is 10 us for each of the 4 chained steps.
    sample = combine_rank_samples(inter, artifact, results(40e-6 * job.timed_iterations))
    assert sample['flow_us'][flow] == pytest.approx(10.0)
    payload = results(None)
    payload[0]['payload_bytes'] = job.payload_bytes
    with pytest.raises(ValueError, match='payload_bytes'):
        combine_rank_samples(job, artifact, payload)


def test_chain_length_is_bounded_by_the_chain_byte_limit(tmp_path):
    from vericcl.verification.online.probe_jobs import plan_probes
    context = probe_context(tmp_path, channels=1, ranks=2)
    for job in plan_probes(context):
        policy = context.inputs.request_config['sketch']['calibration']
        assert job.chain_steps == max(1, min(policy['chain_steps'], policy['chain_max_bytes'] // job.payload_bytes))
        assert job.chain_steps == 1 or job.runtime_payload_bytes <= policy['chain_max_bytes']
