"""Refinement: the incumbent's routes in finer slices (S -> S/factor)."""
import pytest

from vericcl.errors import SemanticError
from vericcl.tuning.resynthesis import ResynthesisRequest, refine_network
from vericcl.xml.v5_model import NetworkOperation


def op(key, src, dst, contributors, deps=(), refs=(), channel=0, time=0.):
    return NetworkOperation(key, 'send', src, dst, channel, 0, frozenset(contributors), frozenset(contributors),
                            (0, 1, 2), frozenset(deps), frozenset(refs), time)


def test_refinement_keeps_routes_and_maps_slices_into_their_byte_ranges():
    # Two ranks, two slices per rank: rank 1's slice 1 is global id 1 * 2 + 1 = 3.
    ops = (op('a', 1, 0, {3}, time=1.), op('b', 0, 2, {3}, deps={'a'}, refs={'a'}, channel=1, time=2.))
    refined = {item.operation_id: item for item in refine_network(ops, 2, 4, 8)}
    assert sorted(refined) == ['a-p%02d' % i for i in range(4)] + ['b-p%02d' % i for i in range(4)]
    for index in range(4):
        a, b = refined['a-p%02d' % index], refined['b-p%02d' % index]
        # Rank 1 owns new positions 4..7 of its 8; old position 1 covers 4..7.
        assert a.contributors == a.target_contributors == frozenset({1 * 8 + 1 * 4 + index})
        assert (a.src_rank, a.dst_rank, a.channel, a.priority_time) == (1, 0, 0, 1.)
        assert (b.src_rank, b.dst_rank, b.channel, b.priority_time) == (0, 2, 1, 2.)
        assert b.dependencies == b.reference_dependencies == frozenset({'a-p%02d' % index})


def test_refinement_requests_need_a_dividing_factor():
    assert ResynthesisRequest('refinement', 'finer', 1024).slice_bytes == 1024
    with pytest.raises(SemanticError):
        ResynthesisRequest('refinement', 'finer')
    with pytest.raises(SemanticError):
        refine_network((), 2, 1, 2)
    with pytest.raises(SemanticError):
        refine_network((), 2, 2, 5)
