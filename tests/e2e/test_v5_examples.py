"""Public v5 examples: complete CLI, independent contributions and roundtrip."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


def solve_and_verify(tmp_path, documents):
    from tools.generate_examples import write_documents
    from tests.e2e.v5_reference import replay_xml_contributions
    collective = documents['sketch']['collective']
    operator, root_rank = collective['operator'], collective.get('root', 0)
    ranks = documents['topology']['ranks']
    slices = documents['sketch']['hyperparameters']['total_size_bytes'] // documents['sketch']['hyperparameters']['slice_size_bytes']
    paths = write_documents(tmp_path / 'input', documents)
    common = ['--project-root', str(tmp_path), '--case', 'matrix', '--run-id', 'solve']
    for name, path in paths.items():
        common += ['--' + name, str(path)]
    command = [sys.executable, '-m', 'vericcl', 'solve', *common]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=45)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    root = tmp_path / 'examples/matrix/output/solve'
    summary = json.loads((root / 'run-summary.json').read_text())
    assert summary['contract_met'] and summary['stop_reason'] in {'maximum_rounds', 'proven_no_improvement'}
    accepted = json.loads((root / 'best/accepted.json').read_text())
    record = accepted['candidate']
    xml, sidecar = root / record['xml'], root / record['sidecar']
    replay_xml_contributions(xml, operator, ranks=ranks, slices=slices, root=root_rank)
    index = common.index('--run-id') + 1; common[index] = 'verify'
    checked = subprocess.run([sys.executable, '-m', 'vericcl', 'verify', *common,
                              '--xml', str(xml), '--sidecar', str(sidecar)],
                             capture_output=True, text=True, timeout=30)
    assert checked.returncode == 0, checked.stdout + checked.stderr


@pytest.mark.parametrize('method', ['hierarchy', 'constructive_trees'])
@pytest.mark.parametrize('operator', ['broadcast', 'reduce', 'allgather', 'allreduce', 'alltoall', 'reduce_scatter'])
@pytest.mark.parametrize('inplace', [False, True])
def test_v5_example_solves_and_verifies_through_public_cli(tmp_path, operator, method, inplace):
    from tools.generate_examples import example_documents
    solve_and_verify(tmp_path, example_documents(operator, method, inplace=inplace))


@pytest.mark.parametrize('name', ['explicit-symmetry', 'explicit-shortest_paths', 'explicit-batching',
                                  'explicit-fused-tb', 'explicit-root-1', 'explicit-manual'])
def test_explicit_optional_modes_reach_checked_xml(tmp_path, name):
    from tools.generate_examples import example_cases
    solve_and_verify(tmp_path, dict(example_cases())[name])


def test_example_generator_uses_defaults_and_is_immutable(tmp_path):
    from tools.generate_examples import example_documents, write_documents
    from vericcl.input.loader import load_request
    documents = example_documents('allreduce', 'hierarchy')
    assert documents['sketch']['collective'] == {'operator': 'allreduce', 'inplace': False}
    paths = write_documents(tmp_path, documents)
    inputs = load_request(paths)
    assert inputs.collective.datatype == 'float32' and inputs.collective.reduction_op == 'sum'
    with pytest.raises(FileExistsError):
        write_documents(tmp_path, documents)
