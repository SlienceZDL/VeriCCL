"""Working records skip per-write fsync; finalization and resume cover power loss."""
import json
import os

import pytest

from tests.integration.test_lowering_v5 import candidate_fixture
from vericcl.execution.context import RunContext, finalize_run
from vericcl.execution.events import emit_event
from vericcl.execution.status import RunResult
from vericcl.xml.lower import lower_and_check


@pytest.fixture
def fsync_calls(monkeypatch):
    calls = []
    real = os.fsync

    def counting(descriptor):
        calls.append(os.readlink('/proc/self/fd/%d' % descriptor))
        return real(descriptor)
    monkeypatch.setattr(os, 'fsync', counting)
    return calls


def test_default_atomic_write_is_durable_and_opt_out_is_not(tmp_path, fsync_calls):
    from vericcl.artifacts.writer import atomic_write_bytes
    atomic_write_bytes(tmp_path / 'durable', b'a')
    assert len(fsync_calls) == 2
    fsync_calls.clear()
    atomic_write_bytes(tmp_path / 'working', b'b', durable=False)
    assert fsync_calls == [] and (tmp_path / 'working').read_bytes() == b'b'
    assert not [path for path in tmp_path.iterdir() if path.name.endswith('.tmp')]


def test_run_records_and_events_are_not_fsynced_per_write(tmp_path, fsync_calls):
    context = RunContext.create(tmp_path, 'durability', 'r')
    context.write_json('models/m.json', {'a': 1})
    context.write_json('checkpoints/latest.json', {'b': 2}, replace=True)
    context.write_json('checkpoints/latest.json', {'b': 3}, replace=True)
    emit_event(context, 'progress', {'step': 1})
    assert fsync_calls == []
    assert json.loads((context.root / 'checkpoints/latest.json').read_text()) == {'b': 3}
    assert (context.root / 'events.jsonl').read_text().count('\n') == 2


def test_finalization_makes_only_the_published_result_durable(tmp_path, fsync_calls):
    context = RunContext.create(tmp_path, 'durability', 'r')
    for index in range(5):
        context.write_json('candidates/c%d/candidate.json' % index, {'index': index})
    context.write_bytes('candidates/c4/schedule.xml', b'<algo/>')
    context.write_json('candidates/c4/sidecar.json', {})
    context.write_json('models/m1.json', {})
    context.write_json('best/accepted.json', {'model_version': 'm1', 'candidate': {
        'record': 'candidates/c4/candidate.json', 'xml': 'candidates/c4/schedule.xml',
        'sidecar': 'candidates/c4/sidecar.json'}})
    context.write_json('checkpoints/s1/state.json', {})
    context.write_json('checkpoints/latest.json', {'state': 'checkpoints/s1/state.json', 'sha256': ''})
    assert fsync_calls == []
    assert finalize_run(context, RunResult(True, 'completed')).complete
    synced = {os.path.relpath(path, context.root) for path in fsync_calls}
    assert {'run.json', 'run-summary.json', 'reports/summary.md', 'events.jsonl', 'best/accepted.json',
            'candidates/c4/candidate.json', 'candidates/c4/schedule.xml', 'candidates/c4/sidecar.json',
            'models/m1.json', 'checkpoints/latest.json', 'checkpoints/s1/state.json'} <= synced
    assert not any(path.startswith('candidates/c0') for path in synced)


def _two_checkpoints(tmp_path):
    from vericcl.workflow_state import save_checkpoint, build_budgets
    candidate, lowering = candidate_fixture(tmp_path, 'broadcast')
    bundle = lower_and_check(candidate, lowering)
    run = RunContext.create(tmp_path, 'resume', 'r')
    budgets = build_budgets(lowering.inputs)
    first = save_checkpoint(run, lowering.inputs, budgets, lowering.model, bundle, phase='initial')
    second = save_checkpoint(run, lowering.inputs, budgets, lowering.model, bundle, phase='tuning')
    latest = json.loads((run.root / 'checkpoints/latest.json').read_text())
    return run, first, second, latest


def test_resume_after_reboot_falls_back_to_newest_verifying_checkpoint(tmp_path, monkeypatch):
    from vericcl import workflow_state
    run, first, second, latest = _two_checkpoints(tmp_path)
    # Power loss dropped unflushed data written only by the newest checkpoint
    # (its incumbent record). Request documents and an unchanged model version
    # are shared with the older checkpoint, which wrote them first.
    assert second['model'] == first['model']
    (run.root / second['bundle']['sidecar']).write_bytes(b'')
    monkeypatch.setattr(workflow_state, '_boot_id', lambda: 'a-later-boot')
    resumed, state, _ = workflow_state.restore_checkpoint(run.root)
    assert state['phase'] == 'initial' and state['saved_wall'] == first['saved_wall']
    event = [item for item in map(json.loads, (run.root / 'events.jsonl').read_text().splitlines())
             if item['kind'] == 'run_resumed'][-1]
    assert event['data']['checkpoint_fallback']['state'] != latest['state']
    assert 'checksum' in event['data']['checkpoint_fallback']['reason']


def test_resume_after_reboot_tolerates_a_lost_latest_pointer(tmp_path, monkeypatch):
    from vericcl import workflow_state
    run, _, second, _ = _two_checkpoints(tmp_path)
    (run.root / 'checkpoints/latest.json').write_bytes(b'')
    monkeypatch.setattr(workflow_state, '_boot_id', lambda: 'a-later-boot')
    _, state, _ = workflow_state.restore_checkpoint(run.root)
    assert state['phase'] == 'tuning' and state['saved_wall'] == second['saved_wall']


def test_same_boot_damage_is_still_refused(tmp_path):
    from vericcl.workflow_state import restore_checkpoint
    run, _, second, _ = _two_checkpoints(tmp_path)
    (run.root / second['bundle']['sidecar']).write_bytes(b'')
    with pytest.raises(ValueError, match='checksum'):
        restore_checkpoint(run.root)


def test_history_index_lost_to_power_failure_reads_as_empty(tmp_path):
    from vericcl.solver.history_index import _load
    path = tmp_path / 'history.json'
    path.write_bytes(b'')
    assert _load(path) == {'schema_version': 5, 'buckets': {}}
