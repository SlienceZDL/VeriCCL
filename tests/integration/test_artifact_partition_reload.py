"""Saved XML keeps the original request identity across automatic S selection."""
import json

import pytest

from tests.unit.input.v5_support import request_documents, write_request
from vericcl.cli.main import main
from vericcl.execution.budget import StandaloneLease
from vericcl.execution.layout import artifact_path, read_run_section
from vericcl.input.loader import load_request
from vericcl.performance.initial import initial_cost_model
from vericcl.planner.parameters import initial_parameter_domain, resolve_parameters
from vericcl.topology import load_topology
from vericcl.xml.lower import LoweringContext, lower_and_check
from vericcl.xml.v5_model import NetworkCandidate, NetworkOperation


def artifact(tmp_path, original_slice=1024):
    documents = request_documents('broadcast', 2)
    documents['sketch']['hyperparameters']['slice_size_bytes'] = original_slice
    documents['sketch']['solver']['max_channels'] = 2
    documents['sketch']['budgets']['max_tuning_rounds'] = 0
    documents['topology']['local_costs'] = [dict(rank=r, copy=dict(alpha_us=.1, beta_us_per_byte=.001))
                                         for r in range(2)]
    files = write_request(tmp_path, documents)
    original = load_request(files)
    choice = next(c for c in initial_parameter_domain(original, fixed_slice_bytes=2048).candidates
                  if c.channels == 1)
    bound = resolve_parameters(original, choice)
    operations = tuple(NetworkOperation('slice-%d' % s, 'send', 0, 1, 0, 0,
        frozenset({s}), frozenset({s}), (0, 1), frozenset(), frozenset(), float(s)) for s in range(2))
    source = NetworkCandidate(2, 2, 2048, operations, 'saved-partition-fixture')
    lowering = LoweringContext(bound, load_topology(bound), initial_cost_model(bound), StandaloneLease.after(10))
    bundle = lower_and_check(source, lowering)
    assert bundle.can_execute
    xml = tmp_path / 'saved.xml'
    sidecar = tmp_path / 'saved-sidecar.json'
    xml.write_text(bundle.xml_text)
    sidecar.write_text(json.dumps(bundle.sidecar))
    args = ['verify', '--project-root', str(tmp_path), '--case', 'reloaded', '--run-id', 'r']
    for name, path in files.items():
        args.extend(['--' + name, str(path)])
    args.extend(['--xml', str(xml), '--sidecar', str(sidecar)])
    return original, args, sidecar


@pytest.mark.parametrize('original_slice', [1024, None])
def test_public_verify_loads_selected_slice_without_rewriting_original_identity(tmp_path, original_slice):
    original, args, _ = artifact(tmp_path, original_slice)
    assert main(args) == 0
    run = tmp_path / 'examples/reloaded/output/r'
    accepted = json.loads(artifact_path(run, 'best/accepted.json').read_text())['candidate']
    assert accepted['input_sha256'] == original.input_sha256
    assert accepted['effective_config']['sketch']['hyperparameters']['slice_size_bytes'] == 2048
    assert accepted['effective_config']['sketch']['solver']['max_channels'] == 2
    request = read_run_section(run, 'effective-request.json')
    assert request['sketch']['hyperparameters']['slice_size_bytes'] == original_slice
    assert all(check['status'] == 'valid' for check in accepted['checks'].values())
    latest = json.loads(artifact_path(run, 'checkpoints/latest.json').read_text())
    checkpoint = json.loads((run / latest['state']).read_text())
    assert checkpoint['input_sha256'] == original.input_sha256
    restored_sketch = json.loads((run / checkpoint['files']['sketch']).read_text())
    assert restored_sketch['hyperparameters']['slice_size_bytes'] == 2048


@pytest.mark.parametrize('slice_bytes', [True, 0, 2048.0, 1536, 1026])
def test_public_verify_rejects_illegal_artifact_partition(tmp_path, slice_bytes):
    _, args, path = artifact(tmp_path)
    sidecar = json.loads(path.read_text())
    sidecar['request_contract']['slice_bytes'] = slice_bytes
    path.write_text(json.dumps(sidecar))
    assert main(args) == 6
    run = tmp_path / 'examples/reloaded/output/r'
    assert not artifact_path(run, 'best/accepted.json').exists()
    summary = json.loads(artifact_path(run, 'run-summary.json').read_text())
    assert summary['failure_category'] == 'invalid'
    records = [json.loads(p.read_text()) for p in artifact_path(run, 'candidates').glob('*/candidate.json')]
    assert len(records) == 1 and records[0]['checks']['xml']['status'] == 'invalid'


@pytest.mark.parametrize('changed', ['budget', 'trace', 'root'])
def test_public_verify_keeps_original_identity_when_binding_slice(tmp_path, changed):
    _, args, _ = artifact(tmp_path)
    path = tmp_path / 'sketch.json'
    data = json.loads(path.read_text())
    if changed == 'budget':
        data['budgets']['max_offline_validation_total_s'] += 1
    elif changed == 'trace':
        # This changes an optional protocol field without enabling hardware.
        data.setdefault('trace_and_measure', {})['seed'] = 7
    else:
        data['collective']['root'] = 1
    path.write_text(json.dumps(data))
    assert main(args) == 6
    run = tmp_path / 'examples/reloaded/output/r'
    assert not artifact_path(run, 'best/accepted.json').exists()
    summary = json.loads(artifact_path(run, 'run-summary.json').read_text())
    assert summary['failure_category'] == 'invalid'
    records = [json.loads(p.read_text()) for p in artifact_path(run, 'candidates').glob('*/candidate.json')]
    assert len(records) == 1 and records[0]['checks']['xml']['status'] == 'invalid'


def test_public_verify_does_not_rebind_other_external_contract_fields(tmp_path):
    _, args, path = artifact(tmp_path)
    sidecar = json.loads(path.read_text())
    sidecar['request_contract']['root'] = 1
    path.write_text(json.dumps(sidecar))
    assert main(args) != 0
    assert not (tmp_path / 'examples/reloaded/output/r/process/best/accepted.json').exists()


def test_public_verify_rechecks_original_byte_ban_after_binding_larger_slice(tmp_path):
    _, args, path = artifact(tmp_path)
    atom_path = tmp_path / 'atom.json'
    atom = json.loads(atom_path.read_text())
    # Original slice 0 is bytes [0, 1024); it remains banned when the XML
    # transfers one larger slice containing bytes [0, 2048).
    atom['forbidden_transfers'] = [[0, 0, 1, 0]]
    atom_path.write_text(json.dumps(atom))
    original = load_request({key: tmp_path / (key + '.json')
                             for key in ('topology', 'sketch', 'atom', 'runtime')})
    sidecar = json.loads(path.read_text())
    # An input digest is not a proof: an untrusted sidecar can claim it.
    sidecar['input_sha256'] = original.input_sha256
    path.write_text(json.dumps(sidecar))
    assert main(args) != 0
    run = tmp_path / 'examples/reloaded/output/r'
    assert not artifact_path(run, 'best/accepted.json').exists()
    records = [json.loads(p.read_text()) for p in artifact_path(run, 'candidates').glob('*/candidate.json')]
    assert len(records) == 1
    assert records[0]['checks']['xml']['status'] == 'valid'
    assert records[0]['checks']['semantic']['status'] == 'invalid'
    assert 'directed-edge' in records[0]['checks']['semantic']['message']
    assert records[0]['effective_config']['sketch']['hyperparameters']['reference_slice_size_bytes'] == 1024


@pytest.mark.parametrize('operator', ['alltoall', 'reduce_scatter'])
def test_artifact_partition_retains_collective_rank_divisibility(tmp_path, operator):
    from vericcl.errors import SemanticError
    from vericcl.planner.parameters import bind_artifact_partition
    inputs = load_request(write_request(tmp_path, request_documents(operator, 4)))
    sidecar = {'input_sha256': inputs.input_sha256, 'request_contract': {'slice_bytes': 2048}}
    with pytest.raises(SemanticError):
        bind_artifact_partition(inputs, sidecar)  # 4096/2048 = 2 slices for four ranks.
    sidecar['request_contract']['slice_bytes'] = 512
    bound = bind_artifact_partition(inputs, sidecar)
    assert bound.hyperparameters.slice_count == 8
    assert bound.input_sha256 == inputs.input_sha256


def test_artifact_boolean_slice_is_rejected_even_when_original_slice_is_one(tmp_path):
    from vericcl.errors import SemanticError
    from vericcl.planner.parameters import bind_artifact_partition
    documents = request_documents('broadcast', 2)
    documents['sketch']['collective']['datatype'] = 'uint8'
    documents['sketch']['hyperparameters'].update(total_size_bytes=4, slice_size_bytes=1)
    inputs = load_request(write_request(tmp_path, documents))
    with pytest.raises(SemanticError):
        bind_artifact_partition(inputs, {'input_sha256': inputs.input_sha256,
                                        'request_contract': {'slice_bytes': True}})


def test_low_level_reload_rejects_same_value_float_slice_contract(tmp_path):
    from vericcl.xml.parser import check_xml_bundle
    original, _, path = artifact(tmp_path)
    choice = next(c for c in initial_parameter_domain(original, fixed_slice_bytes=2048).candidates
                  if c.channels == 1)
    inputs = resolve_parameters(original, choice)
    sidecar = json.loads(path.read_text())
    sidecar['request_contract']['slice_bytes'] = 2048.0
    lowering = LoweringContext(inputs, load_topology(inputs), initial_cost_model(inputs), StandaloneLease.after(10))
    result = check_xml_bundle((tmp_path / 'saved.xml').read_text(), sidecar, lowering)
    assert not result.can_execute
    assert result.checks['xml'].status.value == 'invalid'


@pytest.mark.parametrize('contract', [None, [], 'invalid'])
def test_low_level_reload_reports_malformed_contract_as_invalid(tmp_path, contract):
    from vericcl.xml.parser import check_xml_bundle
    original, _, path = artifact(tmp_path, 2048)
    sidecar = json.loads(path.read_text())
    sidecar['request_contract'] = contract
    lowering = LoweringContext(original, load_topology(original), initial_cost_model(original), StandaloneLease.after(10))
    result = check_xml_bundle((tmp_path / 'saved.xml').read_text(), sidecar, lowering)
    assert not result.can_execute
    assert result.checks['xml'].status.value == 'invalid'
