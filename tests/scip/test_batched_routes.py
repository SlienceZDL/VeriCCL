from fractions import Fraction
from collections import Counter
import itertools

import pytest


def test_partial_extension_preserves_prefix_and_whole_items():
    from vericcl.solver.trees.batch import BatchPacker
    packer = BatchPacker((0, 1, 2), {(0, 1): 2, (1, 2): 1, (0, 2): 1}, {0: ('a', 'b')})
    first = packer.extend(0, 0, 1)
    assert first.amount == 2 and first.remainder_id is None
    split = packer.extend(0, 1, 2)
    assert split.amount == 1 and split.remainder_id is not None
    assert packer.batches[split.remainder_id].edges == ((0, 1),)
    assert set(packer.batches[0].items + packer.batches[split.remainder_id].items) == {'a', 'b'}
    batches = packer.finish()
    assert len(batches) == 2
    assert all(len(batch.edges) == 2 for batch in batches)


def test_zero_capacity_extension_does_not_mutate_state():
    from vericcl.solver.trees.batch import BatchPacker
    packer = BatchPacker((0, 1), {(0, 1): 2}, {0: ('a', 'b')})
    before = dict(packer.batches)
    assert packer.extension_amount(0, 1, 0) == 0
    assert packer.batches == before


def test_residual_cut_keeps_capacity_needed_by_another_root():
    from vericcl.solver.trees.batch import BatchPacker
    packer = BatchPacker((0, 1, 2), {(0, 1): 1, (1, 2): 1, (2, 0): 1, (0, 2): 1},
                         {0: ('a',), 1: ('b',)})
    assert packer.extend(0, 0, 1).amount == 1
    assert packer.extension_amount(0, 1, 2) == 0
    assert packer.extend(0, 0, 2).amount == 1
    assert len(packer.finish()) == 2


def test_infeasible_rooted_cut_is_rejected():
    from vericcl.solver.trees.batch import BatchPacker
    from vericcl.errors import ConstructionInfeasibleError
    with pytest.raises(ConstructionInfeasibleError, match='cut'):
        BatchPacker((0, 1, 2), {(0, 1): 1, (1, 2): 1}, {0: ('a', 'b')})


def test_exact_integer_bandwidth_optimum_matches_brute_force():
    from vericcl.solver.trees.packing import pack_bandwidth
    # Three possible root-0 trees. Compare every allocation of three copies.
    edges = {(0, 1): Fraction(2), (0, 2): Fraction(1), (1, 2): Fraction(3), (2, 1): Fraction(1)}
    trees = (((0, 1), (0, 2)), ((0, 1), (1, 2)), ((0, 2), (2, 1)))
    optimum = None
    for assignment in itertools.product(range(3), repeat=3):
        load = Counter(edge for index in assignment for edge in trees[index])
        value = max(count * edges[edge] for edge, count in load.items())
        optimum = value if optimum is None else min(optimum, value)
    packed = pack_bandwidth((0, 1, 2), edges, {0: ('a', 'b', 'c')})
    assert packed.bandwidth_time_us == optimum
    assert packed.exact_optimal and packed.scope == 'integer_rooted_trees_without_shared_resources'
    load = Counter(edge for batch in packed.batches for edge in batch.edges for _ in batch.items)
    assert max(count * edges[edge] for edge, count in load.items()) == optimum


def test_multi_root_batch_capacity_counts_every_root():
    from vericcl.solver.trees.packing import pack_bandwidth
    result = pack_bandwidth((0, 1), {(0, 1): Fraction(2), (1, 0): Fraction(3)},
                            {0: ('0a', '0b'), 1: ('1a', '1b')})
    assert result.bandwidth_time_us == 6
    assert {batch.root: len(batch.items) for batch in result.batches} == {0: 2, 1: 2}


@pytest.mark.parametrize('seed', range(6))
def test_small_directed_multi_root_optimum_matches_exhaustive_trees(seed):
    import random
    from vericcl.solver.trees.packing import pack_bandwidth
    from vericcl.solver.trees.operators import tree_columns
    rng = random.Random(seed)
    edges = {(a, b): Fraction(rng.randint(1, 5), rng.randint(1, 3))
             for a in range(3) for b in range(3) if a != b}
    choices = [tuple(tree_columns((0, 1, 2), root, edges)) for root in range(3)]
    assert all(len(trees) == 3 for trees in choices)
    best = min(max(Counter(edge for tree in selected for edge in tree)[edge] * duration
                   for edge, duration in edges.items()) for selected in itertools.product(*choices))
    result = pack_bandwidth((0, 1, 2), edges, {r: (str(r),) for r in range(3)})
    assert result.bandwidth_time_us == best


def test_physical_path_changes_topology_identity(tmp_path):
    from dataclasses import replace
    from tests.unit.input.v5_support import request_documents, write_request
    from vericcl.input.loader import load_request
    from vericcl.topology import load_topology
    original = load_topology(load_request(write_request(tmp_path, request_documents('broadcast', 2))))
    links = {key: replace(edge, physical_path=('gpu0', 'switch-new', 'gpu1'))
             for key, edge in original.links.items()}
    changed = replace(original, links=links, isomorphism_signature='')
    assert changed.isomorphism_signature != original.isomorphism_signature


def test_switch_path_restoration_retains_physical_resources(tmp_path):
    from tests.unit.input.v5_support import request_documents, write_request
    from vericcl.input.loader import load_request
    from vericcl.topology import load_topology
    from vericcl.solver.trees.switch_paths import restore_physical_paths
    docs = request_documents('broadcast', 2)
    docs['topology']['directed_links'][0].update(physical_path=['gpu0', 'nic0', 'switch0', 'nic1', 'gpu1'], resources=['uplink'])
    docs['topology']['shared_resources'] = [{'id': 'uplink', 'capacity_bytes_per_us': 100,
                                            'links': [[0, 1]]}]
    topology = load_topology(load_request(write_request(tmp_path, docs)))
    restored = restore_physical_paths(topology, ((0, 1),))
    assert restored[0].physical_path == ('gpu0', 'nic0', 'switch0', 'nic1', 'gpu1')
    assert restored[0].resources == ('uplink',)

