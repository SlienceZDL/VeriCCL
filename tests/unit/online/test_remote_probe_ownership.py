import json
from pathlib import Path

import pytest

from tests.unit.online.test_probe_jobs import probe_context
from vericcl.execution.context import RunContext


def test_remote_cleanup_distinguishes_process_exit_from_gpu_release(tmp_path):
    from vericcl.verification.online.remote import _cleanup
    root = tmp_path / 'launch'; root.mkdir()
    (root / 'prepared-worker-a.json').write_text(json.dumps({'expected_ranks': [0]}))
    rank = root / 'ranks/0'; rank.mkdir(parents=True)
    (rank / 'rank-result.json').write_text(json.dumps({'rank': 0, 'cleanup_complete': True, 'gpu_released': False}))
    result = _cleanup(root, 'worker-a', .1)
    assert result['cleanup_complete'] and not result['gpu_released']
    assert (root / 'STOP').exists()


def test_missing_rank_identity_is_not_a_release_acknowledgement(tmp_path):
    from vericcl.verification.online.remote import _cleanup
    root = tmp_path / 'launch'; root.mkdir()
    (root / 'prepared-worker-a.json').write_text(json.dumps({'expected_ranks': [0]}))
    result = _cleanup(root, 'worker-a', .01)
    assert not result['cleanup_complete'] and not result['gpu_released']


def test_retained_uncertain_gpu_ownership_blocks_only_overlapping_jobs(tmp_path):
    from vericcl.verification.online.remote import NativeProbeDriver
    from vericcl.verification.online.probe_jobs import plan_probes
    context = probe_context(tmp_path, channels=1, ranks=2)
    context.run_context = RunContext.create(tmp_path, 'ownership', 'run')
    context.run_context.write_json('live/native-resource-ownership.json', {'quarantined': [['worker-a', 0]]})
    job = next(j for j in plan_probes(context) if j.category == 'copy' and j.rank_ids == (0,))
    driver = NativeProbeDriver(project_root=tmp_path)
    result = driver.execute(job, context)
    assert result.status == 'blocked_owned_resources'
    assert driver.quarantined == {('worker-a', 0)}
    assert not context.run_context.artifact('models/probes/' + job.job_id + '/probe.xml').exists()


def test_remote_hosts_and_mutable_paths_reject_shell_or_boundary_escape(tmp_path):
    from vericcl.verification.online.remote import _host, owned_home_path
    for name in ('-oProxyCommand=evil', 'worker-a;echo bad', 'example-user@worker-a', 'worker-a\nworker_b'):
        with pytest.raises(ValueError):
            _host(name)
    for path in ('/tmp/VeriCCL', '/home/other-user', '/home/example-user/../other-user', 'relative/path'):
        with pytest.raises(ValueError):
            owned_home_path(path)
    assert owned_home_path(tmp_path) == tmp_path.resolve()
