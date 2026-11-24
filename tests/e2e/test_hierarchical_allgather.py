import pytest

from tests.e2e._support import (
    assert_end_to_end,
    assert_stages_pipelined,
    documents,
    network_pairs,
    stages,
    step_types,
)


pytestmark = pytest.mark.phase07

RANKS = 4
GATEWAYS = {0, RANKS // 2}


def _node(rank):
    return rank // (RANKS // 2)


def _request():
    return documents('allgather', ranks=RANKS, slices=4, method='hierarchy', gateway_nodes=True, stage_num=3)


def test_gateway_hierarchy_composes_exact_global_allgather(solved):
    run = solved(_request())
    assert_end_to_end(run)
    grouped = stages(run)

    assert run.sidecar['version_schedule']['stage_count'] == 3
    assert list(grouped) == [0, 1, 2]
    cross_node = {stage: {tuple(op['participants']) for op in operations
                          if len({_node(rank) for rank in op['participants']}) > 1}
                  for stage, operations in grouped.items()}
    # Gather to the gateway inside each node, exchange between gateways, then
    # distribute inside each node.
    assert cross_node == {0: set(), 1: {(0, 2)}, 2: set()}
    assert {op['destination']['rank'] for op in grouped[0]} <= GATEWAYS
    assert {op['destination']['rank'] for op in grouped[2]} == set(range(RANKS))
    assert all(op['kind'] == 'copy' for operations in grouped.values() for op in operations)
    pairs = network_pairs(run)
    assert {(src, dst) for src, dst in pairs if _node(src) != _node(dst)} == {(0, 2), (2, 0)}
    assert step_types(run) == {'cpy', 'r', 's'}
    assert all((int(tb.attrib['send']) >= 0) != (int(tb.attrib['recv']) >= 0)
               for tb in run.xml.xpath('./gpu/tb') if tb.xpath("./step[@type!='cpy' and @type!='nop']"))
    assert_stages_pipelined(run)


def test_gateway_hierarchy_xml_passes_symbolic_bdd_verification(solved):
    run = solved(_request())
    bundle, symbolic = assert_end_to_end(run)

    operations = len(run.sidecar['version_schedule']['operations'])
    assert symbolic.statistics['operations'] == operations
    assert symbolic.statistics['bdd_nodes'] > 0
    # The verdict persisted by the run equals the one recomputed from the XML.
    persisted = run.record['checks']['semantic']['evidence']['symbolic_bdd']
    assert persisted == dict(symbolic.checks)
    assert bundle.checks['semantic'].status.value == 'valid'
