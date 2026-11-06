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
    metadata = json.loads((root / 'run.json').read_text())
    summary = metadata['result']
    assert summary['contract_met'] and summary['stop_reason'] in {'maximum_rounds', 'proven_no_improvement'}
    accepted = metadata['best']
    record = accepted['candidate']
    xml, sidecar = root / 'schedule.xml', root / 'schedule.json'
    assert xml.read_bytes() == (root / record['xml']).read_bytes()
    assert sidecar.read_bytes() == (root / record['sidecar']).read_bytes()
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


@pytest.mark.parametrize('method', ['hierarchy', 'auto'])
def test_public_solve_honors_manual_stage_edge_ban(tmp_path, method):
    from tools.generate_examples import example_documents, write_documents
    from vericcl.cli.main import main
    from vericcl.input.loader import load_request
    from vericcl.planner.direct import build_direct_plan
    from vericcl.topology import load_topology
    documents = example_documents('allgather', 'hierarchy', ranks=2)
    documents['sketch']['hyperparameters'].update(total_size_bytes=1024, slice_size_bytes=1024)
    original = load_request(write_documents(tmp_path / 'automatic', documents))
    plan = build_direct_plan(original, load_topology(original))
    def states(interface):
        return [{'rank': slot.rank, 'offset': slot.offset, 'contributors': sorted(value)}
                for slot, value in interface.values.items()]
    nodes = []
    for node in plan.nodes:
        source_zero = any(0 in value for value in node.logical_input.values.values())
        nodes.append({'id': node.node_id, 'stage_id': 1 if source_zero else 0,
            'operator': node.local_collective.kind.value, 'root': node.local_collective.root,
            'group': list(node.communication_group), 'input_states': states(node.logical_input),
            'output_states': states(node.logical_output),
            'atoms': [{'slice_ids': [0], 'src_ranks': [0], 'dst_ranks': [1]}] if source_zero else []})
    documents['atom'].update(synthesis_method=method, M_switch=1,
                            manual={'enabled': True, 'nodes': nodes})
    paths = write_documents(tmp_path / 'manual', documents)
    arguments = ['solve', '--project-root', str(tmp_path), '--case', 'manual', '--run-id', 'r']
    for key, path in paths.items():
        arguments += ['--' + key, str(path)]
    assert main(arguments) == 3
    root = tmp_path / 'examples/manual/output/r'
    metadata = json.loads((root / 'run.json').read_text())
    assert metadata['result']['contract_met'] is False
    assert metadata['best'] is None
    assert not (root / 'schedule.xml').exists()
