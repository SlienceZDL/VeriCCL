from dataclasses import replace

from tests.unit.online.test_probe_jobs import probe_context
from vericcl.verification.online.probe_jobs import plan_probes


def test_avg_component_failure_does_not_disable_send_or_copy_measurement(tmp_path):
    context = probe_context(tmp_path, channels=1, ranks=2)
    context.inputs = replace(context.inputs, collective=replace(context.inputs.collective, reduction_op='avg'))
    jobs = list(plan_probes(context))
    independent = [job for job in jobs if all(flow.kind in {'send', 'copy'} for flow in job.flows)]
    reducing = [job for job in jobs if any(flow.kind in {'rrc', 'reduce'} for flow in job.flows)]
    assert independent and reducing
    # send/copy perform no reduction. Their executable kernel parameter and
    # cache identity must be independent of the application's reduction op.
    assert {job.reduction_op for job in independent} == {'sum'}
    assert {job.reduction_op for job in reducing} == {'avg'}
    assert context.inputs.collective.reduction_op == 'avg'
