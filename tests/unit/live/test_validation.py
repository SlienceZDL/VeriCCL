from dataclasses import replace
import pytest


def test_persistent_samples_require_quiescent_work_and_separate_retained_resources():
    from vericcl.verification.online.acceptance import LaunchSample, LaunchRequest, AcceptanceContext, _validate_sample
    sample = LaunchSample(10, True, False, False, 'sample', 'native_msccl_live', 'identity', (),
                          gpu_work_quiescent=True, resources_retained=True)
    _validate_sample(sample, LaunchRequest('sample', 1, 1, 1, 'A'), AcceptanceContext(True))
    with pytest.raises(Exception, match='unresolved'):
        _validate_sample(replace(sample, gpu_work_quiescent=False), LaunchRequest('sample', 1, 1, 1, 'A'), AcceptanceContext(True))


def test_prepared_live_trace_storage_preserves_normal_light_telemetry(tmp_path):
    from tests.integration.test_lowering_v5 import candidate_fixture
    from vericcl.xml.lower import lower_and_check
    from vericcl.live.manifest import export_manifest
    candidate, context = candidate_fixture(tmp_path, 'broadcast')
    bundle = lower_and_check(candidate, context)
    manifest = export_manifest(bundle, context.inputs, tmp_path / 'schedule.xml',
        trace_records={0: 128, 1: 128}, trace_prefix=tmp_path / 'trace', live_trace=True)
    assert manifest['light_telemetry'] and all(rank['trace_records'] == 128 for rank in manifest['ranks'])
