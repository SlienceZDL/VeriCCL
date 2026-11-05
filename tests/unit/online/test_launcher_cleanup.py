import base64
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from tests.unit.online.test_probe_jobs import probe_context


@pytest.mark.parametrize('launcher_released', [False, True])
def test_probe_rank_release_does_not_replace_launcher_cleanup(tmp_path, monkeypatch, launcher_released):
    from vericcl.execution.context import RunContext
    from vericcl.input.schema import _freeze
    from vericcl.verification.online import remote
    from vericcl.verification.online.probe_jobs import plan_probes
    from vericcl.verification.online.probe_xml import build_probe_xml
    from vericcl.workflow_state import plain

    context = probe_context(tmp_path, channels=1, ranks=2)
    context.run_context = RunContext.create(tmp_path, 'probe-launcher-cleanup', 'run')
    config = plain(context.inputs.request_config)
    config['runtime']['paths'] = dict(vericcl_driver='/unused/probe', mpi_bin='/unused/mpi')
    context.inputs = replace(context.inputs, request_config=_freeze(config))
    job = next(job for job in plan_probes(context) if job.category == 'copy')
    driver = remote.NativeProbeDriver(project_root=tmp_path)
    monkeypatch.setattr(driver, '_is_local', lambda host: True)
    monkeypatch.setattr(driver, '_call', lambda *a, **kw: SimpleNamespace(
        status='timeout', cleanup_complete=False, as_json=lambda: dict(status='timeout', returncode=None)))
    monkeypatch.setattr(remote, 'reconcile_owned_process', lambda *a: launcher_released, raising=False)

    def helper(ctx, host, args, *a):
        if args[0] == 'collect':
            files = {}
            for mapping in job.rank_map:
                if mapping['host'] == host:
                    rank = mapping['rank']
                    receipt = dict(rank=rank, cleanup_complete=True, gpu_released=True)
                    files['ranks/%d/rank-result.json' % rank] = base64.b64encode(json.dumps(receipt).encode()).decode()
            return dict(host=host, files=files)
        return {}

    monkeypatch.setattr(driver, '_helper', helper)
    sample, evidence = driver._launch(job, build_probe_xml(job), context, 1, 1, 60)
    assert sample is None and evidence['error']
    assert evidence['gpu_released'] is True
    assert evidence['cleanup_confirmed'] is launcher_released
