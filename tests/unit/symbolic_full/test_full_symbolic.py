"""Full symbolic execution + BDD mode (sketch.symbolic.enabled)."""
import xml.etree.ElementTree as ET

import pytest

from tests.unit.input.v5_support import request_documents, write_request
from vericcl.input.loader import load_request
from vericcl.topology import load_topology
from vericcl.symbolic_full.checks import CHECKS, full_check
from vericcl.symbolic_full.config import enabled, settings
from vericcl.symbolic_full.executor import FullExecution, LoweringLink
from vericcl.symbolic_full.xmlprog import parse_xml


def ag_inputs(tmp_path, ranks=2, *, symbolic=True, nodes=None, forbidden=None):
    docs = request_documents('allgather', ranks)
    docs['sketch']['collective']['inplace'] = True
    if symbolic:
        docs['sketch']['symbolic'] = {'enabled': True}
    if nodes is not None:
        docs['topology']['nodes'] = nodes
    if forbidden is not None:
        docs['atom']['forbidden_transfers'] = forbidden
    inputs = load_request(write_request(tmp_path, docs))
    return inputs, load_topology(inputs)


def direct_ag(ranks, n, *, max_bytes=2 ** 32):
    """In-place AllGather: every rank sends its n chunks to every peer directly."""
    root = ET.Element('algo', name='t', proto='Simple', coll='allgather', inplace='1', ngpus=str(ranks),
                      nchunksperloop=str(ranks * n), nchannels='1', minBytes='0', maxBytes=str(max_bytes))
    for rank in range(ranks):
        gpu = ET.SubElement(root, 'gpu', id=str(rank), i_chunks=str(n), o_chunks=str(ranks * n), s_chunks='0')
        tb = 0
        for peer in range(ranks):
            if peer == rank:
                continue
            send = ET.SubElement(gpu, 'tb', id=str(tb), send=str(peer), recv='-1', chan='0')
            for k in range(n):
                ET.SubElement(send, 'step', s=str(k), type='s', srcbuf='i', srcoff=str(k), dstbuf='o',
                              dstoff=str(rank * n + k), cnt='1', depid='-1', deps='-1', hasdep='0')
            recv = ET.SubElement(gpu, 'tb', id=str(tb + 1), send='-1', recv=str(peer), chan='0')
            for k in range(n):
                ET.SubElement(recv, 'step', s=str(k), type='r', srcbuf='o', srcoff=str(peer * n + k), dstbuf='o',
                              dstoff=str(peer * n + k), cnt='1', depid='-1', deps='-1', hasdep='0')
            tb += 2
    return root


def step(root, rank, tb, index):
    return root.find("gpu[@id='%d']/tb[@id='%d']/step[@s='%d']" % (rank, tb, index))


def check(root, inputs, topology):
    result, execution = full_check(ET.tostring(root, encoding='unicode'), None, None, None, inputs, topology)
    return result, execution


# -- switch -----------------------------------------------------------------------

def test_switch_is_off_and_absent_by_default(tmp_path):
    inputs, _ = ag_inputs(tmp_path, symbolic=False)
    assert 'symbolic' not in inputs.request_config['sketch']
    assert not enabled(inputs) and settings(inputs).enabled is False


def test_standard_gate_is_used_when_the_switch_is_off(tmp_path):
    from tests.scip.test_joint_repair_v5 import repair_fixture
    bundle, _ = repair_fixture(tmp_path)
    statistics = bundle.checks['semantic'].evidence['symbolic_statistics']
    assert 'mode' not in statistics


# -- BDD reasoning -------------------------------------------------------------------

def test_targeted_reachability_equals_the_transitive_closure():
    import random
    from vericcl.symbolic_full.space import FullSpace
    rng = random.Random(7)
    s = FullSpace(((('node', 40), ('node2', 40), ('node3', 40)),))
    for _ in range(6):
        edges = {(a, b) for a in range(40) for b in range(a + 1, 40) if rng.random() < .06}
        relation = s.relation(('node', 'node2'), edges)
        need = s.relation(('node', 'node2'), {(rng.randrange(40), rng.randrange(40)) for _ in range(80)})
        closure = s.closure(relation, left=('node',), right=('node2',), middle=('node3',))
        assert s.unreached(need, relation, left='node', right='node2', middle='node3') == need & ~closure


def test_peeling_orders_a_dag_and_isolates_a_cycle():
    from vericcl.symbolic_full.space import FullSpace
    s = FullSpace(((('node', 6), ('node2', 6)),))
    nodes = s.domain('node')
    layers, rest = s.peel(nodes, s.relation(('node', 'node2'), {(0, 1), (1, 2), (0, 3)}), present=('node',), nxt=('node2',))
    assert [s.values(layer, 'node') for layer in layers] == [(0, 4, 5), (1, 3), (2,)] and s.is_empty(rest)
    _, rest = s.peel(nodes, s.relation(('node', 'node2'), {(0, 1), (1, 2), (2, 1)}), present=('node',), nxt=('node2',))
    assert s.values(rest, 'node') == (1, 2)


# -- correctness -------------------------------------------------------------------

def test_a_correct_xml_passes_every_full_check(tmp_path):
    inputs, topology = ag_inputs(tmp_path, ranks=3)
    result, execution = check(direct_ag(3, 4), inputs, topology)
    assert result.valid, result.witnesses
    assert set(result.checks) == set(CHECKS)
    # Symbols and atoms exist only through the symbolic execution; every
    # received chunk is one Symbol and one {s, pd, t} atom with its path.
    assert len(execution.symbols) == len(execution.transfers) == 3 * 2 * 4
    atom = next(iter(execution.atoms.values()))
    assert atom.pd.symbols and atom.pd.symbols[-1].dst_rank == int(atom.state_id[1:].split(':')[0])
    assert all(entry['deadlock'] is False for entry in result.statistics['progress'])


def test_fifo_capacity_deadlock_appears_only_at_large_messages(tmp_path):
    inputs, topology = ag_inputs(tmp_path)
    root = direct_ag(2, 4)
    for rank in (0, 1):
        # The receive threadblock waits for the third send, which needs a free
        # FIFO slot that only the peer's first receive releases.
        step(root, rank, 0, 2).set('hasdep', '1')
        first = step(root, rank, 1, 0)
        first.set('depid', '0'), first.set('deps', '2')
    result, _ = check(root, inputs, topology)
    assert result.checks['deadlock'] == 'invalid'
    witness = result.witnesses['deadlock']
    assert witness['fifo_messages'] == 2 and witness['cycle']
    # Small messages leave enough FIFO slots: the same program makes progress.
    root.set('maxBytes', '64')
    assert check(root, inputs, topology)[0].checks['deadlock'] == 'valid'


def test_dependency_cycle_is_a_deadlock_with_a_cycle_witness(tmp_path):
    inputs, topology = ag_inputs(tmp_path)
    root = direct_ag(2, 4)
    for a, b in ((0, 1), (1, 0)):
        step(root, 0, b, 1).set('hasdep', '1')
        target = step(root, 0, a, 0)
        target.set('depid', str(b)), target.set('deps', '1')
    result, execution = check(root, inputs, topology)
    assert not result.valid and result.checks['deadlock'] == 'invalid'
    assert not execution.order, 'no execution order exists for a cyclic program'


def test_unordered_writes_to_one_region_are_a_buffer_race(tmp_path):
    inputs, topology = ag_inputs(tmp_path)
    root = direct_ag(2, 4)
    gpu = root.find("gpu[@id='0']")
    extra = ET.SubElement(gpu, 'tb', id='2', send='-1', recv='-1', chan='0')
    ET.SubElement(extra, 'step', s='0', type='cpy', srcbuf='i', srcoff='0', dstbuf='o', dstoff='4', cnt='1',
                  depid='-1', deps='-1', hasdep='0')
    result, _ = check(root, inputs, topology)
    assert result.checks['buffer_safety'] == 'invalid'


def test_a_wrong_destination_breaks_exactly_once_at_the_output(tmp_path):
    inputs, topology = ag_inputs(tmp_path)
    root = direct_ag(2, 4)
    step(root, 1, 1, 0).set('dstoff', '1')
    result, _ = check(root, inputs, topology)
    assert result.checks['exactly_once'] == 'invalid'
    assert result.witnesses['exactly_once']['output_rank'] == 1


def test_connection_sequences_must_match(tmp_path):
    inputs, topology = ag_inputs(tmp_path)
    root = direct_ag(2, 4)
    step(root, 1, 1, 3).set('cnt', '2')
    result, _ = check(root, inputs, topology)
    assert result.checks['fifo_order'] == 'invalid'


def test_forbidden_bytes_are_found_by_the_bdd(tmp_path):
    inputs, topology = ag_inputs(tmp_path, forbidden=[[0, 0, 1, 0]])
    result, _ = check(direct_ag(2, 4), inputs, topology)
    assert result.checks['constraints'] == 'invalid'
    assert result.witnesses['constraints'] == {'slice': 0, 'src': 0, 'dst': 1, 'stage': 0}


def test_relays_between_nodes_must_use_gateways(tmp_path):
    nodes = [{'id': 0, 'ranks': [0, 1], 'gateways': [0]}, {'id': 1, 'ranks': [2], 'gateways': [2]}]
    inputs, topology = ag_inputs(tmp_path, ranks=3, nodes=nodes)
    result, _ = check(direct_ag(3, 2), inputs, topology)
    assert result.checks['relay_legality'] == 'invalid'
    assert result.witnesses['relay_legality'] in ({'src': 1, 'dst': 2}, {'src': 2, 'dst': 1})


# -- lowering gate ------------------------------------------------------------------

@pytest.fixture
def full_fixture(tmp_path, monkeypatch):
    """The repair fixture (slow 0->1 link, idle 0->2) with the switch on."""
    import tests.scip.test_joint_repair_v5 as fixture_module
    original = fixture_module.request_documents

    def documents(*args, **kwargs):
        docs = original(*args, **kwargs)
        docs['sketch']['symbolic'] = {'enabled': True}
        return docs
    monkeypatch.setattr(fixture_module, 'request_documents', documents)
    bundle, context = fixture_module.repair_fixture(tmp_path)
    assert enabled(context.lowering.inputs)
    return bundle, context


def test_lowering_gate_uses_the_full_checks_when_on(full_fixture):
    bundle, _ = full_fixture
    evidence = bundle.checks['semantic'].evidence
    assert evidence['symbolic_statistics']['mode'] == 'full'
    assert set(evidence['symbolic_bdd']) == set(CHECKS)
    assert all(value == 'valid' for value in evidence['symbolic_bdd'].values())


# -- analysis and repair ---------------------------------------------------------------

def test_full_hotspot_has_an_idle_alternative_from_a_holder(full_fixture):
    from vericcl.symbolic_full.cache import execution_for
    from vericcl.symbolic_full.hotspots import locate
    bundle, context = full_fixture
    lowering = context.lowering
    execution = execution_for(bundle, lowering.inputs)
    report = locate(execution, bundle, lowering.inputs, lowering.topology, lowering.model)
    assert report.hotspots and report.counters['mode'] == 'full'
    top = report.hotspots[0]
    assert top.congested_resource == 'link:0:1' and (0, 2) in top.alternative_links
    assert top.evidence['evidence_level'] == 'full_symbolic_execution_bdd_counters'
    assert 0 in top.evidence['holders'] and top.related_operations


def test_full_repair_screens_reroutes_as_bdd_decisions(full_fixture):
    from vericcl.symbolic_full.cache import execution_for
    from vericcl.symbolic_full.hotspots import locate
    from vericcl.symbolic_full.repair import scope_and_plans
    from vericcl.tuning.bottleneck import BottleneckSettings
    from vericcl.tuning.correctness import _network_from_storage
    from vericcl.verification.hotspots import HotspotSettings
    bundle, context = full_fixture
    lowering = context.lowering
    report = locate(execution_for(bundle, lowering.inputs), bundle, lowering.inputs, lowering.topology, lowering.model)
    scope, plans, evidence = scope_and_plans(report.hotspots[0], bundle, _network_from_storage(bundle), report,
                                             lowering, BottleneckSettings(), HotspotSettings.from_inputs(lowering.inputs))
    assert scope.reroutable[0] == report.hotspots[0].operation_id
    assert plans and evidence['decision_variables'] >= 1 and evidence['legal_assignments'] >= 1
    for items in plans.values():
        for plan in items:
            assert plan.path[-1] == 1 and 'link:0:1' not in {'link:%d:%d' % hop for hop in zip(plan.path, plan.path[1:])}


def test_full_bottleneck_repair_runs_end_to_end(full_fixture):
    from vericcl.symbolic_full.cache import execution_for
    from vericcl.symbolic_full.hotspots import locate
    from vericcl.tuning.bottleneck import bottleneck_strategies, solve_bottleneck
    bundle, context = full_fixture
    lowering = context.lowering
    report = locate(execution_for(bundle, lowering.inputs), bundle, lowering.inputs, lowering.topology, lowering.model)
    family, strategy = next(iter(bottleneck_strategies(report, bundle, lowering)))
    result = solve_bottleneck(bundle, strategy, report, context)
    full = next(item['bottleneck']['full_symbolic'] for item in result.solver_evidence if 'bottleneck' in item)
    assert full['mode'] == 'full'
    if result.candidate is not None:
        assert result.candidate.can_execute
        assert result.candidate.checks['semantic'].evidence['symbolic_statistics']['mode'] == 'full'


def test_threadblock_order_hint_runs_ready_data_first(tmp_path):
    from vericcl.performance.initial import initial_cost_model
    from vericcl.symbolic_full.order import tb_order_hints
    inputs, topology = ag_inputs(tmp_path, ranks=3)
    # rank 1 forwards rank 0's chunk to rank 2 and then sends its own chunk:
    # its own chunk is ready at once but waits behind the forwarded one.
    root = ET.Element('algo', name='t', proto='Simple', coll='allgather', inplace='1', ngpus='3',
                      nchunksperloop='12', nchannels='1', minBytes='0', maxBytes='4294967296')
    def tb(rank, tb_id, send, recv, steps):
        gpu = root.find("gpu[@id='%d']" % rank)
        if gpu is None:
            gpu = ET.SubElement(root, 'gpu', id=str(rank), i_chunks='4', o_chunks='12', s_chunks='0')
        block = ET.SubElement(gpu, 'tb', id=str(tb_id), send=str(send), recv=str(recv), chan='0')
        for index, (kind, sb, so, db, do, dep, hasdep) in enumerate(steps):
            ET.SubElement(block, 'step', s=str(index), type=kind, srcbuf=sb, srcoff=str(so), dstbuf=db, dstoff=str(do),
                          cnt='1', depid=str(dep[0]), deps=str(dep[1]), hasdep=str(hasdep))
    tb(0, 0, 1, -1, [('s', 'i', 0, 'o', 0, (-1, -1), 0)])
    tb(1, 0, -1, 0, [('r', 'o', 0, 'o', 0, (-1, -1), 1)])
    tb(1, 1, 2, -1, [('s', 'o', 0, 'o', 0, (0, 0), 0), ('s', 'i', 0, 'o', 4, (-1, -1), 0)])
    tb(2, 0, -1, 1, [('r', 'o', 0, 'o', 0, (-1, -1), 0), ('r', 'o', 4, 'o', 4, (-1, -1), 0)])
    program = parse_xml(ET.tostring(root, encoding='unicode'))
    names = {(b.rank, b.tb, s.index): 'op-%d-%d-%d' % (b.rank, b.tb, s.index) for b in program.blocks for s in b.steps}
    link = LoweringLink(names, {}, {}, {})
    execution = FullExecution(program, inputs, link, settings(inputs)).run()
    execution.bind(initial_cost_model(inputs))
    hints = tb_order_hints(execution, inputs)
    pairs = {record.orders[0] for record, _ in hints}
    assert ('op-1-1-1', 'op-1-1-0') in pairs, pairs
    assert all(gain > 0 for _, gain in hints)
