from dataclasses import replace
from collections import Counter

from tests.unit.input.v5_support import request_documents, write_request
from vericcl.input.loader import load_request
from vericcl.topology import load_topology
from vericcl.execution.budget import StandaloneLease


def probe_context(tmp_path, *, channels=3, ranks=3):
    from vericcl.verification.online.probe_jobs import ProbeContext
    documents = request_documents('allreduce', ranks)
    documents['sketch']['solver']['max_channels'] = channels
    documents['sketch']['calibration'].update(enabled=True, max_relative_fit_error=.1, job_timeout_s=30)
    documents['sketch']['budgets'].update(max_online_validation_per_round_s=120, max_online_validation_total_s=300)
    inputs = load_request(write_request(tmp_path, documents))
    return ProbeContext(inputs, load_topology(inputs), StandaloneLease.after(30))


def test_planner_covers_all_directed_links_and_each_allowed_channel_count(tmp_path):
    from vericcl.verification.online.probe_jobs import plan_probes
    context = probe_context(tmp_path)
    jobs = list(plan_probes(context))
    for edge in context.topology.links:
        for kind in ('send', 'rrc'):
            for count in (1, 2, 3):
                matches = [job for job in jobs if job.category == 'link' and len(job.flows) == count
                           and all(flow.kind == kind and (flow.src_rank, flow.dst_rank) == (edge.src_rank, edge.dst_rank)
                                   for flow in job.flows)]
                assert len({job.payload_bytes for job in matches}) >= 3
                assert context.inputs.hyperparameters.slice_size_bytes in {job.payload_bytes for job in matches}
                assert all({flow.channel for flow in job.flows} == set(range(count)) for job in matches)
    assert len({job.job_id for job in jobs}) == len(jobs)


def test_local_copy_reduce_jobs_are_separate_from_communication(tmp_path):
    from vericcl.verification.online.probe_jobs import plan_probes
    context = probe_context(tmp_path)
    jobs = list(plan_probes(context))
    assert {job.category for job in jobs} >= {'link', 'copy', 'reduce'}
    for job in jobs:
        if job.category in {'copy', 'reduce'}:
            assert len(job.flows) == 1
            assert job.flows[0].src_rank == job.flows[0].dst_rank
        assert job.datatype == 'float32' and job.reduction_op == 'sum'
        assert job.launches == 20 and job.warmup_iterations == 5 and job.timed_iterations == 20


def test_requested_multiflow_modes_are_deduplicated_and_separate(tmp_path):
    from vericcl.verification.online.probe_jobs import plan_probes, ProbeFlow
    context = probe_context(tmp_path)
    flows = tuple(ProbeFlow('send', 0, 1, c) for c in range(3))
    context.observed_modes = (flows, tuple(reversed(flows)))
    jobs = list(plan_probes(context))
    modes = [job for job in jobs if tuple(flow.flow_id for flow in job.flows) == tuple(flow.flow_id for flow in flows)]
    assert len(modes) == len({job.payload_bytes for job in modes})


def test_probe_sizes_do_not_change_application_partition(tmp_path):
    from vericcl.verification.online.probe_jobs import plan_probes
    context = probe_context(tmp_path)
    original = context.inputs.hyperparameters
    jobs = list(plan_probes(context))
    assert len({job.payload_bytes for job in jobs}) >= 3
    assert context.inputs.hyperparameters is original
    assert all(job.payload_bytes % 4 == 0 for job in jobs)


def test_host_scope_and_active_rank_mapping_preserve_original_identity(tmp_path):
    from vericcl.verification.online.probe_jobs import plan_probes
    context = probe_context(tmp_path)
    config = {key: dict(value) if isinstance(value, dict) else value for key, value in context.inputs.request_config.items()}
    # Rank identities survive job-local dense MPI remapping; host identity is
    # taken from explicit runtime placement, not guessed from topology groups.
    jobs = list(plan_probes(context))
    for job in jobs:
        assert set(job.rank_ids) == {rank for flow in job.flows for rank in (flow.src_rank, flow.dst_rank)}
        assert job.host_scope in {'local', 'intra', 'inter'}


def test_shared_pairs_and_actual_three_flow_modes_have_separate_jobs(tmp_path):
    from vericcl.verification.online.probe_jobs import ProbeContext, ProbeFlow, plan_probes
    documents = request_documents('allreduce', 3)
    documents['sketch']['solver']['max_channels'] = 2
    documents['sketch']['calibration'].update(enabled=True, max_relative_fit_error=.1, job_timeout_s=30)
    documents['sketch']['budgets'].update(max_online_validation_per_round_s=120, max_online_validation_total_s=300)
    documents['topology']['nodes'] = [{'id': 0, 'ranks': [0, 1]}, {'id': 1, 'ranks': [2]}]
    documents['runtime']['rank_map'][2]['host'] = 'worker-c'
    documents['topology']['shared_resources'] = [{'id': 'egress0', 'links': [[0, 1], [0, 2]],
                                                  'capacity_bytes_per_us': 100}]
    for edge in documents['topology']['directed_links']:
        if edge['src'] == 0:
            edge['resources'] = ['egress0']
    inputs = load_request(write_request(tmp_path, documents))
    actual = (ProbeFlow('send', 0, 1, 0), ProbeFlow('send', 0, 1, 1), ProbeFlow('send', 0, 2, 0))
    context = ProbeContext(inputs, load_topology(inputs), StandaloneLease.after(30), observed_modes=(actual,))
    jobs = list(plan_probes(context))
    shared = [j for j in jobs if j.category == 'shared']
    assert {len(j.flows) for j in shared} == {2, 3, 4}
    assert all(j.host_scope == 'inter' and 'egress0' in j.resources for j in shared)
    assert any(j.category == 'link' and j.host_scope == 'intra' for j in jobs)
    assert all({m['original_rank'] for m in j.rank_map} == {0, 1, 2} for j in shared)
