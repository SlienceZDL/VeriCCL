import pytest

from tests.e2e._support import (
    assert_end_to_end,
    assert_stages_pipelined,
    dependency_type_pairs,
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


def _cross_node(operation):
    return len({_node(rank) for rank in operation['participants']}) > 1


def test_gateway_hierarchy_composes_exact_global_allreduce(solved):
    run = solved(documents('allreduce', ranks=RANKS, slices=4, method='hierarchy',
                           gateway_nodes=True, stage_num=4))
    assert_end_to_end(run)
    grouped = stages(run)
    operations = [op for stage in grouped.values() for op in stage]

    assert run.sidecar['version_schedule']['stage_count'] == 4
    assert list(grouped) == [0, 1, 2, 3]
    # Intra-node reduction into each gateway.
    for gateway in GATEWAYS:
        assert any(op['kind'] == 'reduce' and op['destination']['rank'] == gateway and not _cross_node(op)
                   and len(op['participants']) > 1 for op in operations)
    # Only the gateways talk across nodes, both to reduce and to distribute.
    inter = [op for op in operations if _cross_node(op)]
    assert {tuple(sorted(op['participants'])) for op in inter} == {(0, 2)}
    assert {op['kind'] for op in inter} == {'reduce', 'copy'}
    # Each gateway broadcasts the result back inside its node.
    for gateway in GATEWAYS:
        assert any(op['kind'] == 'copy' and gateway in op['participants'] and not _cross_node(op)
                   and op['destination']['rank'] not in GATEWAYS for op in operations)
    pairs = network_pairs(run)
    assert {(src, dst) for src, dst in pairs if _node(src) != _node(dst)} == {(0, 2), (2, 0)}
    assert {'r', 'rrc', 's'} <= step_types(run)
    assert all((int(tb.attrib['send']) >= 0) != (int(tb.attrib['recv']) >= 0)
               for tb in run.xml.xpath('./gpu/tb') if tb.xpath("./step[@type!='cpy' and @type!='nop']"))
    # A reduced chunk is forwarded only after its reduce-receive completed.
    assert ('s', 'rrc') in dependency_type_pairs(run)
    assert_stages_pipelined(run)
