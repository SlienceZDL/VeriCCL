from tests.unit.online.test_probe_jobs import probe_context
from vericcl.input.loader import load_request
from vericcl.input.schema import DATATYPE_BYTES
from vericcl.verification.online.probe_jobs import ProbeContext, payload_sizes, plan_probes, probe_buffsize, probe_environment
from vericcl.xml.capacity import simple_payload_capacity


def _with_runtime_buffsize(tmp_path, buffsize):
    """The fixture request with a runtime NCCL_BUFFSIZE that holds less than one slice."""
    from tests.unit.online.test_probe_jobs import request_documents, write_request, load_topology, StandaloneLease
    documents = request_documents('allreduce', 2)
    documents['sketch']['solver']['max_channels'] = 1
    documents['sketch']['calibration'].update(enabled=True, max_relative_fit_error=.1, job_timeout_s=30)
    documents['sketch']['budgets'].update(max_online_validation_per_round_s=120, max_online_validation_total_s=300)
    documents['runtime']['environment']['NCCL_BUFFSIZE'] = str(buffsize)
    inputs = load_request(write_request(tmp_path, documents))
    return ProbeContext(inputs, load_topology(inputs), StandaloneLease.after(30))


def test_probes_raise_the_buffer_so_one_step_holds_twice_the_slice(tmp_path):
    (tmp_path / 'base').mkdir(); (tmp_path / 'small').mkdir()
    base = probe_context(tmp_path / 'base', channels=1, ranks=2)
    size = base.inputs.hyperparameters.slice_size_bytes
    element = DATATYPE_BYTES[base.inputs.collective.datatype]
    # A Simple step spans 4 of the 8 buffer slots, so a buffer of S bytes
    # carries only S/2 per step and cannot hold one slice.
    context = _with_runtime_buffsize(tmp_path / 'small', size)
    small = size
    assert simple_payload_capacity(small, element) < size
    buffsize = probe_buffsize(context.inputs)
    assert simple_payload_capacity(buffsize, element) >= 2 * size
    assert simple_payload_capacity(buffsize - 8 * element, element) < 2 * size
    sizes = payload_sizes(context)
    assert size in sizes and 2 * size in sizes and len(sizes) >= 3
    assert all(job.payload_bytes <= simple_payload_capacity(buffsize, element) for job in plan_probes(context))


def test_probe_environment_changes_only_the_buffer_and_keeps_a_larger_runtime_value(tmp_path):
    (tmp_path / 'small').mkdir(); (tmp_path / 'large').mkdir()
    context = _with_runtime_buffsize(tmp_path / 'small', 1 << 16)
    runtime = dict(context.inputs.request_config['runtime']['environment'])
    probe = probe_environment(context.inputs)
    assert {k: v for k, v in probe.items() if k != 'NCCL_BUFFSIZE'} == {k: v for k, v in runtime.items() if k != 'NCCL_BUFFSIZE'}
    assert int(probe['NCCL_BUFFSIZE']) == probe_buffsize(context.inputs)
    large = _with_runtime_buffsize(tmp_path / 'large', 1 << 30)
    assert probe_buffsize(large.inputs) == 1 << 30
