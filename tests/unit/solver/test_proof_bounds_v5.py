import pytest


def test_exact_dual_certifies_a_scoped_bound_and_model_changes_invalidate_it():
    from vericcl.solver.proof_bounds import verify_lp_dual
    # min x+y; x+2y>=3; x>=0,y>=0. y=1.5 attains the exact lower bound.
    bound = verify_lp_dual(matrix=((1, 2),), rhs=(3,), objective=(1, 1), dual=('1/2',),
                           model_version='m0', scope_id='routes-A')
    assert bound.value_ratio == (3, 2)
    assert bound.can_prune((3, 2), model_version='m0', scope_id='routes-A')
    assert not bound.can_prune((2, 1), model_version='m0', scope_id='routes-A')
    assert not bound.can_prune((1, 1), model_version='m1', scope_id='routes-A')
    assert not bound.can_prune((1, 1), model_version='m0', scope_id='routes-B')


def test_infeasible_dual_or_primal_value_cannot_be_promoted_to_a_bound():
    from vericcl.errors import SemanticError
    from vericcl.solver.proof_bounds import verify_lp_dual
    with pytest.raises(SemanticError, match='dual'):
        verify_lp_dual(matrix=((1, 2),), rhs=(3,), objective=(1, 1), dual=(1,),
                       model_version='m0', scope_id='routes-A')


def test_v5_throughput_relaxation_drops_startup_for_a_sound_capacity_upper_bound():
    from vericcl.topology.model import PerformanceCurve
    from vericcl.solver.lower_bounds import _maximum_capacity
    curve = PerformanceCurve(100, 100.01, {}, beta_us_per_byte=.01)
    assert _maximum_capacity(curve, 1000, 7) == 100
