"""A bounded complete measurement must allow communicator setup beyond 10 s."""
from dataclasses import replace
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize('job_timeout_s', [180., 8.])
def test_communicator_prepare_uses_job_budget_without_extending_cleanup(tmp_path, monkeypatch, job_timeout_s):
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
    config['sketch']['trace_and_measure']['job_timeout_s'] = job_timeout_s
    inputs = replace(lowering.inputs, request_config=_freeze(config))
    run = RunContext.create(tmp_path, 'prepare-budget', 'one')
    driver = native.NativeScheduleDriver(run, inputs, project_root=tmp_path)
    monkeypatch.setattr(native, 'owned_home_path', lambda value: Path(value))
    monkeypatch.setattr(driver, '_control', lambda _: ('127.0.0.1', '127.0.0.1', 12345))

    def stop_before_transport(*args, **kwargs):
        raise RuntimeError('CPU-only: stop before remote preparation or GPU ownership')

    monkeypatch.setattr(driver.transport, '_helper', stop_before_transport)
    root, *_ = driver._launch(bundle, inputs, launch_id='budget-fixture', warmup=5, timed=20,
                             lease=StandaloneLease.after(300))
    payload = json.loads((root / 'incoming-payload.json').read_text())
    limits = payload['native_config']['timeouts']
    assert payload['timeout_s'] == job_timeout_s
    assert 0 < limits['version_prepare_timeout_s'] < payload['timeout_s']
    assert limits['close_cleanup_timeout_s'] <= 10
    assert limits['control_phase_timeout_s'] <= 10
    if job_timeout_s == 180:
        # A 12 s initialization fits a 180 s job but exceeded the previous
        # hidden 10 s setup cap, before the actual collective could execute.
        assert limits['version_prepare_timeout_s'] > 12
        assert limits['version_prepare_timeout_s'] + 20 + limits['close_cleanup_timeout_s'] < job_timeout_s
