"""Late OS cleanup is separate from the immutable execution result and GPU state."""
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='Linux birth identities and process groups')


def test_late_group_exit_reconciles_only_the_owned_process_cleanup(tmp_path):
    from vericcl.execution.context import RunContext
    from vericcl.execution import processes

    context = RunContext.create(tmp_path, 'process-reconciliation', 'run')
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
    try:
        identity = processes.ProcessIdentity.capture(process.pid)
        result = SimpleNamespace(job_id='original-timeout', identity=identity, cleanup_complete=False)
        context.active_jobs.add(result.job_id)
        original = context.write_json('jobs/original-timeout/result.json', dict(status='timeout',
                                     returncode=None, cleanup_complete=False, gpu_released=False))
        old_bytes = original.read_bytes()
        assert processes.reconcile_owned_process(context, result) is False
        assert result.job_id in context.active_jobs
        process.terminate()
        process.wait(timeout=5)
        assert processes.reconcile_owned_process(context, result) is True
        assert result.job_id not in context.active_jobs
        assert original.read_bytes() == old_bytes
        receipt = json.loads(context.artifact('jobs/original-timeout/cleanup-reconciliation.json').read_text())
        assert receipt['cleanup_complete'] is True
        assert receipt['scope'] == 'owned_local_process_group'
        assert 'gpu_released' not in receipt
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
