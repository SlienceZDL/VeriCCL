import json
from pathlib import Path

from tests.integration.test_primary_lifecycle_v5 import request


def arguments(tmp_path, command, case):
    return [command, '--project-root', str(tmp_path), '--case', case, '--run-id', 'r',
            '--topology', str(tmp_path / 'topology.json'), '--sketch', str(tmp_path / 'sketch.json'),
            '--atom', str(tmp_path / 'atom.json'), '--runtime', str(tmp_path / 'runtime.json')]


def test_cli_solve_inspect_and_verify_use_one_case_contract(tmp_path, capsys):
    from vericcl.cli.main import main
    request(tmp_path)
    assert main(arguments(tmp_path, 'solve', 'cli')) == 0
    root = tmp_path / 'examples/cli/output/r'
    metadata = json.loads((root / 'run.json').read_text())
    summary = metadata['result']
    assert summary['contract_met'] and summary['stop_reason'] == 'maximum_rounds'
    invocation = metadata['invocation']
    assert invocation['argv'][1:] == ['-m', 'vericcl', *arguments(tmp_path, 'solve', 'cli')]
    assert main(['inspect', '--run', str(root)]) == 0
    accepted = metadata['best']
    candidate = accepted['candidate']
    args = arguments(tmp_path, 'verify', 'check') + ['--xml', str(root / candidate['xml']),
                                                  '--sidecar', str(root / candidate['sidecar'])]
    assert main(args) == 0
    assert 'verification_complete' in capsys.readouterr().out


def test_cli_requires_four_explicit_files_and_rejects_old_online_switch(tmp_path, capsys):
    from vericcl.cli.main import main
    assert main(['solve', '--online']) == 2
    assert main(['solve', '--topology', 'a', '--sketch', 'b', '--atom', 'c']) == 2


def test_cli_override_keeps_original_snapshot_and_records_effective_default(tmp_path):
    from vericcl.cli.main import main
    request(tmp_path)
    args = arguments(tmp_path, 'solve', 'override') + ['--set', 'sketch.collective.root=1']
    assert main(args) == 0  # root was omitted in the raw request
    root = tmp_path / 'examples/override/output/r'
    assert 'root' not in json.loads((root / 'process/input_snapshot/sketch.json').read_text())['collective']
    assert json.loads((root / 'run.json').read_text())['effective_request']['sketch']['collective']['root'] == 1
