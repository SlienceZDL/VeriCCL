import threading
import time

import pytest


def lease(seconds=4):
    from vericcl.execution.budget import StandaloneLease
    return StandaloneLease.after(seconds, cleanup_reserve_s=.1)


def create():
    from vericcl.solver.scip_api import ScipAdapter
    _, model = ScipAdapter.create_model('contract')
    return model


def test_native_integer_model_matches_independent_enumeration():
    from vericcl.solver.scip_api import ScipAdapter
    model = create()
    x, y = model.addVar('x', vtype='I', lb=0, ub=5), model.addVar('y', vtype='I', lb=0, ub=5)
    model.addCons(2*x + 3*y >= 7)
    model.setObjective(4*x + 2*y, 'minimize')
    result = ScipAdapter.solve(model, lease())
    expected = min(4*a + 2*b for a in range(6) for b in range(6) if 2*a + 3*b >= 7)
    assert result.status == 'optimal'
    assert result.primal == expected == result.dual_bound
    assert result.evidence_scope == 'numerical_model'
    assert result.relative_gap == 0
    model.freeProb()


def test_indicator_equalities_and_boolean_and_preserve_their_semantics():
    from vericcl.solver.scip_api import ScipAdapter
    model = create()
    a, b, both = (model.addVar(name, vtype='B') for name in ('a', 'b', 'both'))
    x = model.addVar('x', lb=0, ub=20)
    model.addConsAnd([a, b], both)
    ScipAdapter.indicator(model, both, True, x == 7)
    model.addCons(a == 1)
    model.addCons(b == 1)
    model.setObjective(x)
    assert ScipAdapter.solve(model, lease()).primal == 7
    model.freeProb()


def test_infeasible_and_expired_without_incumbent_are_distinct():
    from vericcl.solver.scip_api import ScipAdapter
    model = create()
    x = model.addVar('x', lb=0, ub=1)
    model.addCons(x >= 2)
    assert ScipAdapter.solve(model, lease()).status == 'infeasible'
    model.freeProb()
    model = create()
    model.addVar('x')
    expired = lease(.11)
    time.sleep(.02)
    result = ScipAdapter.solve(model, expired)
    assert result.status == 'time_limit' and result.primal is None
    model.freeProb()


def test_complete_warm_start_accepts_inactive_indicator_with_positive_slack():
    from vericcl.solver.scip_api import ScipAdapter
    from vericcl.execution.budget import StandaloneLease
    _, model = ScipAdapter.create_model('indicator-seed')
    try:
        ScipAdapter.configure(model, lease=StandaloneLease.after(5))
        x = model.addVar('x', lb=0, ub=10)
        y = model.addVar('y', lb=0, ub=10)
        active = model.addVar('active', vtype='B')
        ScipAdapter.indicator(model, active, True, x <= y)
        model.setObjective(x)
        assert ScipAdapter.warm_start(model, [(x, 10), (y, 0), (active, 0)], complete=True)
    finally:
        model.freeProb()


def test_complete_warm_start_survives_a_budget_expiring_before_optimize():
    from vericcl.solver.scip_api import ScipAdapter
    model = create()
    x = model.addVar('x', vtype='I', lb=0, ub=10)
    model.addCons(x >= 3)
    model.setObjective(x)
    assert ScipAdapter.warm_start(model, [(x, 5)], complete=True)
    expired = lease(.11)
    time.sleep(.02)
    result = ScipAdapter.solve(model, expired)
    assert result.status == 'time_limit' and result.primal == 5
    model.freeProb()


def test_lexicographic_objectives_keep_primary_bound_separate():
    from vericcl.solver.scip_api import ScipAdapter
    model = create()
    x = model.addVar('x', vtype='I', lb=0, ub=10)
    y = model.addVar('y', vtype='I', lb=0, ub=10)
    model.addCons(x + y >= 5)
    model.set_lexicographic_objectives(((x + y, 'sum'), (x, 'x')))
    result = ScipAdapter.solve(model, lease())
    assert result.primal == 5 and result.dual_bound == 5
    assert result.objective_values == (5, 0)
    assert result.lexicographic_complete
    model.freeProb()


def test_cancelled_model_does_not_begin_optimization():
    from vericcl.solver.scip_api import ScipAdapter
    model = create()
    model.addVar('x')
    budget = lease()
    budget.cancel()
    assert ScipAdapter.solve(model, budget).status == 'cancelled'
    model.freeProb()


def test_thread_caps_and_missing_required_backend_are_explicit(monkeypatch):
    from vericcl.solver.scip_api import ScipAdapter
    from vericcl.errors import SolverUnavailableError
    model = create()
    effective = ScipAdapter.configure(model, seed=3, threads=2, gap=0)
    assert effective['allocated_threads'] == 2
    assert model.getParam('parallel/maxnthreads') <= 2
    model.freeProb()
    monkeypatch.setattr(ScipAdapter, 'available', staticmethod(lambda: False))
    with pytest.raises(SolverUnavailableError, match='SCIP'):
        ScipAdapter.require()


def test_model_construction_uses_the_same_lease_and_returns_owned_state():
    from vericcl.execution.budget import BudgetExhausted
    from vericcl.solver.scip_api import ScipAdapter
    model = create()
    budget = lease(.11)
    ScipAdapter.configure(model, lease=budget)
    time.sleep(.02)
    with pytest.raises(BudgetExhausted):
        model.addVar('too-late')
    model.freeProb()
    assert model._closed


def test_real_solver_event_cancellation_preserves_a_distinct_status():
    from pyscipopt import Eventhdlr, SCIP_EVENTTYPE, quicksum
    from vericcl.solver.scip_api import ScipAdapter
    model = create()
    budget = lease()
    class CancelOnSolution(Eventhdlr):
        def eventinit(self):
            self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)
        def eventexit(self):
            self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)
        def eventexec(self, event):
            budget.cancel()
            self.model.interruptSolve()
    variables = [model.addVar('x' + str(i), vtype='B') for i in range(24)]
    model.addCons(quicksum(variables) >= 12)
    model.setObjective(quicksum((i + 1) * variable for i, variable in enumerate(variables)))
    model.includeEventhdlr(CancelOnSolution(), 'cancel-contract', 'cancel the active native solve')
    result = ScipAdapter.solve(model, budget)
    assert result.status == 'cancelled'
    model.freeProb()
