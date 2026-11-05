from dataclasses import replace
import xml.etree.ElementTree as ET

import pytest

from tests.unit.online.test_probe_jobs import probe_context


def test_probe_xml_uses_independent_slots_and_matching_endpoints(tmp_path):
    from vericcl.verification.online.probe_jobs import plan_probes
    from vericcl.verification.online.probe_xml import build_probe_xml
    context = probe_context(tmp_path, channels=3, ranks=2)
    job = next(j for j in plan_probes(context) if len(j.flows) == 3 and j.flows[0].kind == 'rrc')
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
    job = next(j for j in plan_probes(context) if j.flows[0].kind == 'rrc')
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
