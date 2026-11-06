"""The supported execution boundary has no runtime schedule-update controls."""
import hashlib
import json

import pytest

from tests.unit.input.v5_support import request_documents, write_request
from vericcl.errors import InputValidationError
from vericcl.input.loader import load_request


def test_cli_rejects_removed_live_command():
    from vericcl.cli.main import build_parser
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args(['live', '--topology', 't.json', '--sketch', 's.json',
            '--atom', 'a.json', '--runtime', 'r.json', '--case', 'example',
            '--requests', '1', '--session-timeout-s', '30'])
    assert error.value.code == 2


def test_normalized_request_does_not_add_live_settings(tmp_path):
    inputs = load_request(write_request(tmp_path))
    assert 'live' not in inputs.request_config['sketch']
    assert not any(key.startswith('sketch.live') for key in inputs.value_sources)


@pytest.mark.parametrize('use_override', [False, True])
def test_enabled_live_has_an_actionable_migration_error(tmp_path, use_override):
    docs = request_documents()
    if not use_override:
        docs['sketch']['live'] = {'enabled': True}
    files = write_request(tmp_path, docs)
    with pytest.raises(InputValidationError, match='live.*removed.*solve.*verify'):
        load_request(files, {'sketch.live.enabled': True} if use_override else None)


def test_disabled_legacy_live_is_removed_without_changing_execution_settings(tmp_path):
    docs = request_documents()
    docs['sketch']['live'] = {'enabled': False, 'max_retries': 2}
    docs['runtime']['control'] = {'bind_address': '127.0.0.1', 'port': 31000,
                                  'unix_socket': str(tmp_path / 'obsolete.sock')}
    inputs = load_request(write_request(tmp_path, docs))
    assert 'live' not in inputs.request_config['sketch']
    assert dict(inputs.request_config['runtime']['control']) == {'bind_address': '127.0.0.1', 'port': 31000}
    assert inputs.request_config['sketch']['budgets']['max_tuning_rounds'] == 2


def test_standalone_removed_unix_socket_is_rejected(tmp_path):
    docs = request_documents()
    docs['runtime']['control'] = {'unix_socket': str(tmp_path / 'obsolete.sock')}
    with pytest.raises(InputValidationError, match='unix_socket'):
        load_request(write_request(tmp_path, docs))


def test_single_execution_manifest_never_requests_light_telemetry(tmp_path):
    from tests.integration.test_lowering_v5 import candidate_fixture
    from vericcl.execution.manifest import export_manifest
    from vericcl.xml.lower import lower_and_check
    candidate, lowering = candidate_fixture(tmp_path, 'broadcast')
    bundle = lower_and_check(candidate, lowering)
    manifest = export_manifest(bundle, lowering.inputs, tmp_path / 'schedule.xml')
    assert 'light_telemetry' not in manifest


@pytest.mark.parametrize('action', ['stage-version', 'live-capture-inventory'])
def test_remote_helper_rejects_removed_runtime_update_actions(tmp_path, action, capsys):
    from vericcl.verification.online.remote import main
    xml = '<algo/>'
    version = hashlib.sha256(xml.encode()).hexdigest()
    destination = tmp_path / 'versions' / version
    destination.mkdir(parents=True)
    (tmp_path / 'update.json').write_text(json.dumps({'xml': xml, 'manifest': {
        'version_id': version, 'xml_path': str(destination / 'schedule.xml')}}))
    (tmp_path / 'prepared-worker-a.json').write_text(json.dumps({
        'expected_ranks': [], 'binary_sha256': 'test', 'inventory': {}}))
    with pytest.raises(SystemExit) as error:
        main([action, '--root', str(tmp_path), '--host', 'worker-a',
              '--payload-file', 'update.json', '--capture-id', 'command-1'])
    assert error.value.code == 2
    assert "invalid choice: '" + action + "'" in capsys.readouterr().err
