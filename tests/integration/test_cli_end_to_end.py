"""Public v5 CLI: solve, then verify (and verify --tune) the accepted XML."""
import json

import pytest

from tests.e2e._support import (
    STOP_REASONS,
    assert_accepted,
    cli_arguments,
    documents,
    load_run,
    run_root,
    sha256,
    write_inputs,
)
from vericcl.cli.main import main


pytestmark = pytest.mark.phase07


def _inputs(tmp_path, **options):
    request = documents('allreduce', tuning_rounds=1, **options)
    return request, write_inputs(tmp_path / 'input', request)


def test_public_cli_solves_then_verifies_accepted_xml(tmp_path, capsys):
    request, paths = _inputs(tmp_path)

    assert main(cli_arguments('solve', paths, tmp_path, 'cli', 'solve')) == 0
    assert capsys.readouterr().err == ''
    solved = load_run(run_root(tmp_path, 'cli', 'solve'), request, paths)
    assert_accepted(solved)
    assert solved.summary['timing']['initial_synthesis_s'] > 0.0
    assert solved.summary['timing']['verification_s'] > 0.0
    invocation = json.loads((solved.root / 'invocation.json').read_text())
    assert invocation['command'] == 'solve'

    checked = ('--xml', solved.xml_path, '--sidecar', solved.sidecar_path)
    assert main(cli_arguments('verify', paths, tmp_path, 'cli', 'verify', *checked)) == 0
    assert capsys.readouterr().err == ''
    verified = load_run(run_root(tmp_path, 'cli', 'verify'), request, paths)
    summary = verified.summary
    assert summary['contract_met'] and summary['stop_reason'] == 'verification_complete'
    assert summary['timing']['initial_synthesis_s'] == 0.0
    assert summary['timing']['verification_s'] > 0.0
    assert summary['timing']['tuning_rounds'] == 0
    assert verified.record['origin'] == 'verify_input'
    assert verified.accepted['xml_sha256'] == solved.accepted['xml_sha256']
    assert sha256(verified.root / 'input_snapshot' / 'schedule.xml') == sha256(solved.xml_path)
    assert ({name: check['status'] for name, check in verified.record['checks'].items()}
            == {name: check['status'] for name, check in solved.record['checks'].items()})

    assert main(cli_arguments('verify', paths, tmp_path, 'cli', 'verify-tune', *checked, '--tune')) == 0
    assert capsys.readouterr().err == ''
    tuned = load_run(run_root(tmp_path, 'cli', 'verify-tune'), request, paths)
    assert_accepted(tuned)
    assert tuned.summary['stop_reason'] in STOP_REASONS
    assert tuned.summary['timing']['initial_synthesis_s'] == 0.0
    assert tuned.summary['timing']['tuning_rounds'] <= 1
    # Tuning may only replace the verified XML by a strictly better one.
    assert tuned.accepted['t_pred_us'] <= verified.accepted['t_pred_us']


def test_set_override_changes_effective_request_only(tmp_path, capsys):
    request, paths = _inputs(tmp_path)
    original = paths['sketch'].read_bytes()
    total = 4 * request['sketch']['hyperparameters']['slice_size_bytes']
    override = ('--set', 'sketch.hyperparameters.total_size_bytes=%d' % total)

    # Replacing an explicit value requires --override-input.
    assert main(cli_arguments('solve', paths, tmp_path, 'cli', 'refused', *override)) == 2
    assert capsys.readouterr().err
    assert not run_root(tmp_path, 'cli', 'refused').exists()

    assert main(cli_arguments('solve', paths, tmp_path, 'cli', 'override', *override, '--override-input')) == 0
    assert capsys.readouterr().err == ''
    run = load_run(run_root(tmp_path, 'cli', 'override'), request, paths)
    assert run.summary['contract_met']
    effective = json.loads((run.root / 'effective-request.json').read_text())
    assert effective['sketch']['hyperparameters']['total_size_bytes'] == total
    assert run.record['effective_config']['sketch']['hyperparameters']['total_size_bytes'] == total
    assert run.sidecar['request_contract']['input_bytes'] == total
    snapshot = json.loads((run.root / 'input_snapshot' / 'sketch.json').read_text())
    assert snapshot['hyperparameters']['total_size_bytes'] == request['sketch']['hyperparameters']['total_size_bytes']
    assert paths['sketch'].read_bytes() == original
