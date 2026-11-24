"""Shared helpers for the v5 end-to-end tests.

Every test writes the four v5 JSON documents, runs the public
``python -m vericcl`` CLI and then re-checks the accepted XML independently:
an XML structural contract, the sidecar's declared outputs, a replay of the
XML contributions (``tests.e2e.v5_reference``), the v5 bundle checks and the
symbolic + BDD verdicts. Calibration and trace measurement stay disabled, so
nothing here touches a GPU.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from lxml import etree

from tests.e2e.v5_reference import expected_outputs, replay_xml_contributions
from tests.unit.input.v5_support import request_documents, write_request


PROJECT_ROOT = Path(__file__).parents[2]
OPERATORS = ('broadcast', 'reduce', 'allgather', 'allreduce', 'alltoall', 'reduce_scatter')
SLICE_BYTES = 1024
STOP_REASONS = {'maximum_rounds', 'proven_no_improvement'}
REQUIRED_CHECKS = ('semantic', 'storage', 'endpoint', 'deadlock', 'xml', 'launch', 'prediction')
CLI_TIMEOUT_S = 180


def gateway_topology(ranks):
    """Two nodes; only the first rank of each node has an inter-node link."""
    half = ranks // 2
    gateways = (0, half)
    node = lambda rank: rank // half
    links = []
    for src in range(ranks):
        for dst in range(ranks):
            if src == dst:
                continue
            if node(src) == node(dst):
                links.append({'src': src, 'dst': dst, 'alpha_us': 1.0, 'beta_us_per_byte': 0.01})
            elif {src, dst} == set(gateways):
                links.append({'src': src, 'dst': dst, 'alpha_us': 2.0, 'beta_us_per_byte': 0.03})
    return {'name': 'two-node-gateway', 'ranks': ranks,
            'nodes': [{'id': 0, 'ranks': list(range(half)), 'gateways': [gateways[0]]},
                      {'id': 1, 'ranks': list(range(half, ranks)), 'gateways': [gateways[1]]}],
            'directed_links': links}


def documents(operator, *, ranks=2, slices=2, inplace=False, method='constructive_trees',
              gateway_nodes=False, stage_num=None, seed=0, tuning_rounds=0):
    """Small CPU-only v5 request; ``slices`` is M/S per rank with S = 1 KiB."""
    docs = request_documents(operator, ranks)
    if gateway_nodes:
        docs['topology'] = gateway_topology(ranks)
    docs['topology']['local_costs'] = [
        {'rank': rank, 'copy': {'alpha_us': 0.1, 'beta_us_per_byte': 0.001},
         'reduce': {'alpha_us': 0.2, 'beta_us_per_byte': 0.002}} for rank in range(ranks)]
    sketch = docs['sketch']
    sketch['collective']['inplace'] = inplace
    sketch['hyperparameters'].update(total_size_bytes=slices * SLICE_BYTES, slice_size_bytes=SLICE_BYTES)
    sketch['solver'].update(per_model_timeout_s=5, max_channels=1, max_threads_per_model=1,
                            max_parallel_models=1, solver_seed=seed)
    sketch['budgets'].update(max_initial_synthesis_s=30, max_offline_validation_per_round_s=10,
                             max_offline_validation_total_s=30, max_repair_per_round_s=10,
                             max_repair_total_s=30, max_tuning_rounds=tuning_rounds)
    sketch['calibration'] = {'enabled': False}
    sketch['trace_and_measure'] = {'enabled': False}
    docs['atom'] = {'synthesis_method': method}
    if stage_num is not None:
        docs['atom']['stage_num'] = stage_num
    return docs


def write_inputs(directory, docs):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return write_request(directory, docs)


def cli_arguments(command, paths, project_root, case, run_id, *extra):
    arguments = [command, '--project-root', str(project_root), '--case', case, '--run-id', run_id]
    for name in ('topology', 'sketch', 'atom', 'runtime'):
        arguments += ['--' + name, str(paths[name])]
    return [*arguments, *map(str, extra)]


def run_root(project_root, case, run_id):
    return Path(project_root) / 'examples' / case / 'output' / run_id


@dataclass
class Run:
    root: Path
    documents: dict
    input_paths: dict
    summary: dict
    accepted: dict
    record: dict
    xml_path: Path
    sidecar_path: Path
    xml_text: str
    sidecar: dict
    xml: object
    stdout: str

    @property
    def operator(self):
        return self.documents['sketch']['collective']['operator']

    @property
    def root_rank(self):
        return self.documents['sketch']['collective'].get('root') or 0

    @property
    def ranks(self):
        return self.documents['topology']['ranks']

    @property
    def hyperparameters(self):
        """The partition the accepted XML was lowered with (S may be re-chosen)."""
        return self.record['effective_config']['sketch']['hyperparameters']

    @property
    def slices(self):
        return self.hyperparameters['total_size_bytes'] // self.hyperparameters['slice_size_bytes']


def load_run(root, docs, paths, stdout=''):
    root = Path(root)
    accepted = json.loads((root / 'best' / 'accepted.json').read_text())
    record = accepted['candidate']
    xml_path, sidecar_path = root / record['xml'], root / record['sidecar']
    xml_text = xml_path.read_text()
    return Run(root, docs, paths, json.loads((root / 'run-summary.json').read_text()), accepted, record,
               xml_path, sidecar_path, xml_text, json.loads(sidecar_path.read_text()),
               etree.fromstring(xml_text.encode()), stdout)


def solve(directory, docs, *, case='e2e', run_id='solve'):
    """Run ``python -m vericcl solve`` and load the accepted candidate."""
    directory = Path(directory)
    paths = write_inputs(directory / 'input', docs)
    completed = subprocess.run([sys.executable, '-m', 'vericcl',
                                *cli_arguments('solve', paths, directory, case, run_id)],
                               cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=CLI_TIMEOUT_S)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return load_run(run_root(directory, case, run_id), docs, paths, completed.stdout)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_status(check):
    return check['status'] if isinstance(check, dict) else check


def assert_accepted(run):
    """The run met its contract and its accepted record names this exact XML."""
    summary = run.summary
    assert summary['contract_met'] and summary['execution'] == 'completed', summary
    assert summary['exit_code'] == 0 and summary['failure_category'] is None
    assert summary['stop_reason'] in STOP_REASONS
    assert summary['acceptance'] == 'accepted'
    assert all(_check_status(summary['checks'][name]) == 'valid' for name in REQUIRED_CHECKS)
    assert _check_status(summary['checks']['compatibility']) in {'valid', 'not_run'}
    record = run.record
    assert record['can_execute']
    assert all(record['checks'][name]['status'] == 'valid' for name in REQUIRED_CHECKS)
    digest = sha256(run.xml_path)
    assert digest == run.accepted['xml_sha256'] == record['xml_sha256'] == run.sidecar['xml_sha256']
    assert record['input_sha256'] == run.sidecar['input_sha256']
    assert run.accepted['model_version'] == run.sidecar['model_version']
    assert (run.root / 'reports' / 'timing.json').is_file()


def expected_chunk_geometry(operator, ranks, slices):
    """(i_chunks, o_chunks of ranks that own outputs, nchunksperloop)."""
    outputs = {'allgather': slices * ranks, 'reduce_scatter': slices // ranks}.get(operator, slices)
    loop = slices * ranks if operator == 'allgather' else slices
    return slices, outputs, loop


def steps(run, kind=None):
    return [step for step in run.xml.xpath('./gpu/tb/step') if kind is None or step.attrib['type'] == kind]


def step_types(run):
    return {step.attrib['type'] for step in steps(run)}


def network_pairs(run):
    """(sender, receiver) pairs of the XML's sending threadblocks."""
    return {(int(gpu.attrib['id']), int(tb.attrib['send']))
            for gpu in run.xml.xpath('./gpu') for tb in gpu.xpath('./tb') if int(tb.attrib['send']) >= 0}


def dependency_type_pairs(run):
    pairs = set()
    for gpu in run.xml.xpath('./gpu'):
        blocks = {int(tb.attrib['id']): tb for tb in gpu.xpath('./tb')}
        for tb in blocks.values():
            for step in tb.xpath('./step'):
                depid = int(step.attrib['depid'])
                if depid >= 0:
                    target = blocks[depid].xpath('./step')[int(step.attrib['deps'])]
                    pairs.add((step.attrib['type'], target.attrib['type']))
    return pairs


def assert_xml_contract(run):
    """Structural invariants of an MSCCL XML program, independent of the solver."""
    root = run.xml
    operator, ranks, slices = run.operator, run.ranks, run.slices
    i_chunks, o_chunks, loop = expected_chunk_geometry(operator, ranks, slices)
    assert root.tag == 'algo'
    assert root.attrib['coll'] == operator
    assert int(root.attrib['ngpus']) == ranks
    assert root.attrib['inplace'] == str(int(bool(run.documents['sketch']['collective'].get('inplace'))))
    assert int(root.attrib['nchunksperloop']) == loop
    channels = int(root.attrib['nchannels'])
    gpus = {int(gpu.attrib['id']): gpu for gpu in root.xpath('./gpu')}
    assert sorted(gpus) == list(range(ranks))
    chunks = {rank: {name: int(gpu.attrib[name + '_chunks']) for name in 'ios'} for rank, gpu in gpus.items()}
    for rank, sizes in chunks.items():
        assert sizes['i'] == i_chunks
        if operator != 'reduce' or rank == run.root_rank:
            assert sizes['o'] == o_chunks
    sends, receives = Counter(), Counter()
    for rank, gpu in gpus.items():
        blocks = {int(tb.attrib['id']): tb for tb in gpu.xpath('./tb')}
        assert sorted(blocks) == list(range(len(blocks)))
        for tb in blocks.values():
            channel, send, recv = int(tb.attrib['chan']), int(tb.attrib['send']), int(tb.attrib['recv'])
            assert 0 <= channel < channels
            assert send != rank and recv != rank
            program = tb.xpath('./step')
            assert [int(step.attrib['s']) for step in program] == list(range(len(program)))
            for step in program:
                kind = step.attrib['type']
                assert int(step.attrib['cnt']) == 1
                depid, deps = int(step.attrib['depid']), int(step.attrib['deps'])
                if depid < 0:
                    assert deps == -1
                else:
                    assert depid in blocks and 0 <= deps < len(blocks[depid].xpath('./step'))
                if kind == 'nop':
                    continue
                owners = {'s': (rank, send), 'r': (recv, rank), 'rrc': (rank, rank), 'cpy': (rank, rank)}[kind]
                for prefix, owner in zip(('src', 'dst'), owners):
                    assert 0 <= int(step.attrib[prefix + 'off']) < chunks[owner][step.attrib[prefix + 'buf']]
                if kind == 's':
                    assert send >= 0
                    sends[rank, send, channel] += 1
                elif kind in {'r', 'rrc'}:
                    assert recv >= 0
                    receives[recv, rank, channel] += 1
    assert sends == receives


def assert_declared_outputs(run):
    """The sidecar's final storage states hold exactly the collective's outputs."""
    states = {value['state_id']: frozenset(value['contributors']) for value in run.sidecar['storage']['values']}
    slice_bytes = run.hyperparameters['slice_size_bytes']
    declared = {}
    for output in run.sidecar['version_schedule']['outputs']:
        assert output['offset_bytes'] % slice_bytes == 0
        declared[output['rank'], output['offset_bytes'] // slice_bytes] = states[output['state_id']]
    assert declared == expected_outputs(run.operator, run.ranks, run.slices, run.root_rank)


def replay(run):
    """Independent interpretation of the XML steps (see tests/e2e/v5_reference.py)."""
    return replay_xml_contributions(run.xml_path, run.operator, ranks=run.ranks, slices=run.slices,
                                    root=run.root_rank)


def recheck(run, *, limits=None):
    """Reload the saved XML + sidecar through the v5 checker, as ``verify`` does."""
    from vericcl.execution.budget import StandaloneLease
    from vericcl.input.loader import load_request
    from vericcl.performance.initial import initial_cost_model
    from vericcl.planner.parameters import bind_artifact_partition
    from vericcl.topology import load_topology
    from vericcl.xml.parser import check_xml_bundle
    from vericcl.xml.v5_model import LoweringContext
    original = load_request(run.input_paths)
    assert original.input_sha256 == run.sidecar['input_sha256']
    inputs = bind_artifact_partition(original, run.sidecar)
    model = initial_cost_model(inputs)
    context = LoweringContext(inputs, load_topology(inputs), model, StandaloneLease.after(60), limits=limits)
    return check_xml_bundle(run.xml_text, run.sidecar, context), inputs, model


def assert_offline_checks(run):
    """v5 bundle checks and symbolic + BDD verdicts hold for the accepted XML."""
    from vericcl.verification.symbolic_check import CHECKS, check_bundle
    bundle, inputs, model = recheck(run)
    statuses = {name: check.status.value for name, check in bundle.checks.items()}
    assert bundle.can_execute, statuses
    assert all(statuses[name] == 'valid' for name in REQUIRED_CHECKS), statuses
    assert statuses['compatibility'] in {'valid', 'not_run'}
    assert bundle.xml_sha256 == run.accepted['xml_sha256']
    assert bundle.prediction.model_version == run.accepted['model_version']
    assert abs(bundle.prediction.t_pred_us - run.accepted['t_pred_us']) <= 1e-6 * max(1.0, run.accepted['t_pred_us'])
    symbolic, _ = check_bundle(bundle, inputs, model)
    assert symbolic.valid, symbolic.message
    assert {name: symbolic.checks[name] for name in CHECKS} == dict.fromkeys(CHECKS, 'valid')
    return bundle, symbolic


def assert_end_to_end(run):
    assert_accepted(run)
    assert_xml_contract(run)
    assert_declared_outputs(run)
    replay(run)
    return assert_offline_checks(run)


def stages(run):
    """Version-schedule operations grouped by logical stage."""
    grouped = {}
    for operation in run.sidecar['version_schedule']['operations']:
        grouped.setdefault(operation['stage_id'], []).append(operation)
    return dict(sorted(grouped.items()))


def assert_stages_pipelined(run):
    """No global barrier: each stage starts before the previous one finishes."""
    grouped = list(stages(run).values())
    for earlier, later in zip(grouped, grouped[1:]):
        assert min(op['start_us'] for op in later) < max(op['end_us'] for op in earlier)
