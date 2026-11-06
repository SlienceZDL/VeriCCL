import json
from pathlib import Path
from zipfile import ZipFile

from vericcl.execution.context import RunContext, finalize_run
from vericcl.execution.status import RunResult


def test_process_layout_and_consolidated_run_metadata(tmp_path):
    source = tmp_path / 'input.json'
    source.write_text('{"example": 1}')
    run = RunContext.create(tmp_path, 'case', 'r', input_files={'topology.json': source})
    assert {p.name for p in run.root.iterdir() if p.is_dir()} == {'process'}
    assert run.artifact('models') == run.root / 'process/models'
    assert run.artifact('process/models') == run.artifact('models')
    for filename, value in [('invocation.json', {'command': 'solve'}),
        ('effective-request.json', {'atom': {}}), ('value-sources.json', {'origin': 'file'}),
        ('size-accounting.json', {'input_bytes_per_rank': 16})]:
        run.write_json(filename, value)
        assert not (run.root / filename).exists()
    metadata = json.loads((run.root / 'run.json').read_text())
    assert metadata['schema_version'] == 2
    assert metadata['invocation']['command'] == 'solve'
    assert metadata['effective_request'] == {'atom': {}}
    assert metadata['value_sources'] == {'origin': 'file'}
    assert metadata['size_accounting']['input_bytes_per_rank'] == 16
    assert metadata['input_snapshot']['files']['topology.json']
    assert (run.root / 'process/input_snapshot/topology.json').read_bytes() == source.read_bytes()


def test_final_results_export_matching_candidate_model_and_process_archive(tmp_path):
    run = RunContext.create(tmp_path, 'case', 'r')
    run.write_json('models/model-a.json', {'version': 'model-a'})
    run.write_bytes('candidates/c1/schedule.xml', b'<algo/>')
    run.write_json('candidates/c1/sidecar.json', {'candidate': 'c1'})
    run.write_json('reports/trace-c1.json', {'bottleneck': [0, 1]})
    run.write_json('best/accepted.json', {'model_version': 'model-a', 't_pred_us': 4,
        'candidate': {'xml': 'process/candidates/c1/schedule.xml',
                      'sidecar': 'process/candidates/c1/sidecar.json'},
        'trace_causality': {'path': 'process/reports/trace-c1.json'}})
    run.write_json('checkpoints/cp/state.json', {'phase': 'finished', 'model': 'process/models/model-a.json'})
    run.write_json('checkpoints/latest.json', {'state': 'process/checkpoints/cp/state.json', 'sha256': 'sample'})
    finished = finalize_run(run, RunResult(True, 'completed', acceptance='accepted', stop_reason='round_limit'))
    assert finished.complete
    assert (run.root / 'schedule.xml').read_bytes() == b'<algo/>'
    assert json.loads((run.root / 'schedule.json').read_text()) == {'candidate': 'c1'}
    assert json.loads((run.root / 'model.json').read_text())['version'] == 'model-a'
    assert json.loads((run.root / 'checkpoint.json').read_text())['phase'] == 'finished'
    assert json.loads((run.root / 'trace.json').read_text())['bottleneck'] == [0, 1]
    assert (run.root / 'report.md').is_file()
    metadata = json.loads((run.root / 'run.json').read_text())
    assert metadata['result']['stop_reason'] == 'round_limit'
    assert metadata['artifacts']['schedule'] == 'schedule.xml'
    assert (run.root / 'process/run-summary.json').is_file()
    with ZipFile(run.root / 'details.zip') as archive:
        assert 'process/candidates/c1/schedule.xml' in archive.namelist()
        events = archive.read('process/events.jsonl').decode()
        assert 'run_finalized' in events


def test_failed_run_publishes_report_and_preserves_rejected_evidence(tmp_path):
    run = RunContext.create(tmp_path, 'case', 'r')
    run.write_json('candidates/rejected/candidate.json', {'can_execute': False})
    result = finalize_run(run, RunResult(False, 'incomplete', stop_reason='no_qualified_baseline', failure_category='invalid'))
    assert result.complete
    assert not (run.root / 'schedule.xml').exists()
    assert (run.root / 'process/candidates/rejected/candidate.json').is_file()
    assert 'no_qualified_baseline' in (run.root / 'report.md').read_text()
    assert not (run.root / 'run-summary.json').exists()


def test_existing_schema_one_run_uses_legacy_paths(tmp_path):
    (tmp_path / 'run.json').write_text(json.dumps({'schema_version': 1}))
    from vericcl.execution.security import Redactor
    run = RunContext(tmp_path, task_id='task', run_id='r', redactor=Redactor())
    assert run.artifact('models/a.json') == tmp_path / 'models/a.json'
    run.write_json('effective-request.json', {'legacy': True})
    assert json.loads((tmp_path / 'effective-request.json').read_text()) == {'legacy': True}


def test_process_symlink_cannot_escape_boundary(tmp_path):
    import pytest
    run = RunContext.create(tmp_path, 'case', 'r')
    outside = tmp_path / 'outside'
    outside.mkdir()
    run.artifact('tmp/escape').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='boundary'):
        run.write_json('tmp/escape/value.json', {})
