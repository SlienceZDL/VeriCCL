import json
from pathlib import Path

from tests.unit.input.v5_support import request_documents, write_request
from vericcl.cli.main import main
from vericcl.execution.status import RunResult
from vericcl.input.schema import LIVE_TIMEOUTS


def test_live_entry_records_invocation_before_initial_failure(tmp_path, monkeypatch):
    from vericcl.cli import live

    documents = request_documents('broadcast', 2)
    documents['sketch']['live'] = {'enabled': True, **{name: 10 for name in LIVE_TIMEOUTS}}
    documents['runtime']['control'] = {
        'bind_address': '127.0.0.1', 'port': 49000,
        'unix_socket': str(tmp_path / 'control.sock'),
    }
    files = write_request(tmp_path, documents)
    monkeypatch.setattr(live.PrimaryWorkflow, 'run', lambda self: RunResult(
        False, 'incomplete', stop_reason='injected_no_initial_candidate', failure_category='offline'))
    args = ['live', '--project-root', str(tmp_path), '--case', 'entry', '--run-id', 'r',
            '--requests', '1', '--session-timeout-s', '30']
    for name, path in files.items():
        args.extend(['--' + name, str(path)])
    assert main(args) == 3
    root = tmp_path / 'examples/entry/output/r'
    invocation = json.loads((root / 'invocation.json').read_text())
    assert invocation['cwd'] == str(Path.cwd().resolve())
    assert invocation['command'] == 'live'
    assert invocation['argv'][1:] == ['-m', 'vericcl', *args]
    assert json.loads((root / 'run-summary.json').read_text())['stop_reason'] == 'injected_no_initial_candidate'
