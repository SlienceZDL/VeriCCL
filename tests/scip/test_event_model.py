import pytest


def test_fixed_event_order_optimizes_intervals_without_repaying_startup():
    from vericcl.execution.budget import StandaloneLease
    from vericcl.performance.events import Executable, WorkOperation
    from vericcl.performance.model import CostModel, ServiceCost
    from vericcl.solver.event_model import EventStep, optimize_event_order
    executable = Executable((WorkOperation('a', 'send', 'wire', 100, flow_id='a', resources=('wire',)),
                             WorkOperation('b', 'send', 'wire', 20, flow_id='b', resources=('wire',), not_before_us=5)))
    model = CostModel('m0', {'wire': (ServiceCost(2, .1),)}, {'wire': 10})
    order = tuple((EventStep(kind, op),) for kind, op in (
        ('start', 'a'), ('active', 'a'), ('start', 'b'), ('active', 'b'), ('finish', 'b'), ('finish', 'a')))
    result = optimize_event_order(executable, model, order, StandaloneLease.after(3, cleanup_reserve_s=.1))
    assert result.solver_result.status == 'optimal'
    assert result.solver_result.primal == pytest.approx(14)
    assert result.prediction.t_pred_us == pytest.approx(14)
    assert result.scope == 'fixed_event_order_numerical'
    assert result.full_search_proven is False


def test_event_order_must_cover_every_physical_operation_transition():
    from vericcl.errors import SemanticError
    from vericcl.execution.budget import StandaloneLease
    from vericcl.performance.events import Executable, WorkOperation
    from vericcl.performance.model import CostModel, ServiceCost
    from vericcl.solver.event_model import EventStep, optimize_event_order
    executable = Executable((WorkOperation('a', 'copy', 'copy', 10),))
    with pytest.raises(SemanticError, match='transition'):
        optimize_event_order(executable, CostModel('m0', {'copy': (ServiceCost(1, 1),)}),
                             ((EventStep('start', 'a'),),), StandaloneLease.after(1))
