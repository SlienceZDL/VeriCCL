import pytest

from vericcl.errors import SemanticError
from vericcl.input.loader import load_request
from vericcl.topology import load_topology
from vericcl.topology import isomorphism
from tests.unit.input.v5_support import request_documents, write_request


def nic_topology(tmp_path, *, nodes=4, per_node=3):
    documents = request_documents(ranks=nodes * per_node)
    topology = documents['topology']
    topology['nodes'] = [{'id': node, 'ranks': list(range(node * per_node, (node + 1) * per_node))}
                         for node in range(nodes)]
    resources = {}
    for link in topology['directed_links']:
        source, target = link['src'] // per_node, link['dst'] // per_node
        if source != target:
            names = ['tx-%d' % source, 'rx-%d' % target]
            link['resources'] = names
            for name in names: resources.setdefault(name, []).append([link['src'], link['dst']])
    topology['shared_resources'] = [{'id': name, 'capacity_bytes_per_us': 12500, 'links': links, 'kind': 'nic'}
                                    for name, links in sorted(resources.items())]
    return load_topology(load_request(write_request(tmp_path, documents)))


def test_identical_external_gpu_roles_do_not_expand_every_permutation(tmp_path, monkeypatch):
    topology = nic_topology(tmp_path)
    original = isomorphism._refine_colors
    calls = 0

    def bounded(colors, edges):
        nonlocal calls
        calls += 1
        assert calls <= 20, 'equivalent external GPU permutations repeat canonical search'
        return original(colors, edges)

    monkeypatch.setattr(isomorphism, '_refine_colors', bounded)
    signature = isomorphism.exact_domain_signature(topology, (0, 3, 6, 9))
    assert len(signature) == 64


def test_orbit_pruning_keeps_the_original_exact_signature(tmp_path, monkeypatch):
    topology = nic_topology(tmp_path, nodes=2)
    signature = isomorphism.exact_domain_signature(topology, (0, 3))
    original = isomorphism._canonical_colored_graph

    def unpruned(colors, edges, *, automorphisms=()):
        return original(colors, edges)

    monkeypatch.setattr(isomorphism, '_canonical_colored_graph', unpruned)
    assert isomorphism.exact_domain_signature(topology, (0, 3)) == signature


@pytest.mark.parametrize('colors,edges', [
    (('a', 'b'), ()),
    (('a', 'a'), ((0, 1, 'directed'),)),
])
def test_unproved_vertex_exchanges_are_rejected(colors, edges):
    with pytest.raises(SemanticError, match='automorphism witness'):
        isomorphism._canonical_colored_graph(colors, edges, automorphisms=((1, 0),))
