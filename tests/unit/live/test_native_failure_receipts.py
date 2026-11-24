"""An MPI launcher can return zero while a native rank reports failure."""
import base64
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('record_case', ['failed', 'missing', 'duplicate', 'error', 'incomplete',
                                        'wrapper_failed', 'wrapper_missing', 'launcher_pending',
                                        'launcher_reconciled', 'archive_failed', 'complete'])
def test_native_records_determine_success_after_launcher_returns_zero(tmp_path, monkeypatch, record_case):
    from tests.integration.test_lowering_v5 import candidate_fixture
    from vericcl.execution.budget import StandaloneLease
    from vericcl.execution.context import RunContext
    from vericcl.input.schema import _freeze
    from vericcl.live import native
    from vericcl.workflow_state import plain
    from vericcl.xml.lower import lower_and_check

    candidate, lowering = candidate_fixture(tmp_path, 'allgather')
    bundle = lower_and_check(candidate, lowering)
    config = plain(lowering.inputs.request_config)
    config['runtime']['paths'] = {'vericcl_native': '/unused/native', 'mpi_bin': '/unused/mpi'}
    config['sketch']['trace_and_measure']['job_timeout_s'] = 30
    inputs = replace(lowering.inputs, request_config=_freeze(config))
    run = RunContext.create(tmp_path, 'native-receipts', record_case)
    driver = native.NativeScheduleDriver(run, inputs, project_root=tmp_path)
    monkeypatch.setattr(native, 'owned_home_path', lambda value: Path(value))
    monkeypatch.setattr(native, '_immutable', lambda path, data: Path(path).write_bytes(data))
    monkeypatch.setattr(driver, '_control', lambda _: ('127.0.0.1', '127.0.0.1', 12345))
    monkeypatch.setattr(driver.transport, '_is_local', lambda _: True)
    launcher_failed = record_case.startswith('launcher_')
    monkeypatch.setattr(driver.transport, '_call', lambda *a, **kw: SimpleNamespace(
        status='failed' if launcher_failed else 'completed', cleanup_complete=not launcher_failed,
        as_json=lambda: dict(status='failed' if launcher_failed else 'completed', returncode=0)))
    monkeypatch.setattr(native, 'reconcile_owned_process',
                        lambda *a: record_case == 'launcher_reconciled', raising=False)

    def helper(context, host, args, *a, **kw):
        files = {}
        if args[0] == 'collect':
            if record_case == 'archive_failed' and args[args.index('--cursor') + 1] != '0':
                raise ValueError('injected bulk archive failure after all release receipts')
            for mapping in config['runtime']['rank_map']:
                if mapping['host'] != host:
                    continue
                rank = mapping['rank']
                def encode(value):
                    return base64.b64encode(json.dumps(value).encode()).decode()
                wrapper = dict(rank=rank, cleanup_complete=True, gpu_released=True,
                               status='completed', returncode=0)
                if record_case == 'wrapper_failed' and rank == 0:
                    wrapper.update(status='failed', returncode=1)
                name = 'cleanup-reconciliation.json' if record_case == 'wrapper_missing' and rank == 0 else 'rank-result.json'
                files['ranks/%d/%s' % (rank, name)] = encode(wrapper)
                if record_case == 'missing' and rank == 1:
                    continue
                files['native/rank-%d.json' % rank] = encode(dict(
                    rank=0 if record_case == 'duplicate' else rank,
                    correct=not (record_case == 'failed' and rank == 0), gpu_released=True,
                    gpu_completed=record_case != 'incomplete',
                    error='close failed after numerical checks' if record_case == 'error' and rank == 0 else None))
        result = dict(host=host, files=files, prepared={})
        if args[0] == 'collect' and record_case == 'archive_failed':
            result.update(next_cursor=1, snapshot={'digest': 'fixed-snapshot'})
        return result

    monkeypatch.setattr(driver.transport, '_helper', helper)
    *_, evidence = driver._launch(bundle, inputs, launch_id='cpu-only', warmup=1, timed=1, lease=StandaloneLease.after(30))
    assert evidence['gpu_released']
    assert evidence['cleanup_confirmed'] is (record_case != 'launcher_pending')
    assert bool(evidence['error']) is (record_case != 'complete')
