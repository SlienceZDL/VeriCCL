import json
import os
import time

import pytest

from vericcl.execution.budget import StandaloneLease
from vericcl.execution.context import RunContext
from vericcl.solver.scip_api import ScipAdapter
from vericcl.solver.worker_scope import isolated_solver_scope


def solve_owned(tmp_path, model, seconds=8):
    run = RunContext.create(tmp_path, 'solver', 'run')
    lease = StandaloneLease.after(seconds)
    ScipAdapter.configure(model, lease=lease)
    with isolated_solver_scope(run, cleanup_deadline=time.monotonic() + seconds + 1):
        outcome = model.optimize()
    return run, outcome


def test_native_solver_result_survives_fresh_child_and_parent_remains_reusable(tmp_path):
    _, model = ScipAdapter.create_model('owned-integer')
    x = model.addVar('x', vtype='INTEGER', lb=0)
    model.addCons(x >= 3)
    model.setObjective(x)
    run, result = solve_owned(tmp_path, model)
    assert result.numerical_optimal and model.getVal(x) == 3
    identity = json.loads(next((run.artifact('jobs')).glob('scip-*/identity.json')).read_text())
    assert identity['pid'] == identity['pgid'] and identity['pid'] != os.getpid()
    assert not run.active_jobs
    assert next((run.artifact('tmp')).glob('scip-*/model.cip')).is_file()
    model.freeProb()


def test_lexicographic_objectives_and_parent_expression_values_survive(tmp_path):
    _, model = ScipAdapter.create_model('owned-lex')
    x = model.addVar('x', vtype='INTEGER', lb=0)
    y = model.addVar('y', vtype='INTEGER', lb=0)
    model.addCons(x + y >= 3)
    model.set_lexicographic_objectives(((x + y + 2, 'completion'), (y, 'traffic')))
    _, result = solve_owned(tmp_path, model)
    assert result.lexicographic_complete and result.objective_values == (5, 0)
    assert model.getVal(x + y + 2) == 5
    model.freeProb()


def test_indicator_and_and_constraint_survive_cip_with_complete_incumbent(tmp_path):
    _, model = ScipAdapter.create_model('owned-indicator')
    a, b, c = [model.addVar(name, vtype='BINARY') for name in ('a', 'b', 'c')]
    x = model.addVar('x', vtype='INTEGER', lb=0)
    model.addCons(a == 1)
    model.addCons(b == 1)
    model.addConsAnd([a, b], c)
    ScipAdapter.indicator(model, c, True, x >= 7)
    model.setObjective(x)
    assert ScipAdapter.warm_start(model, [(a, 1), (b, 1), (c, 1), (x, 7)], complete=True)
    _, result = solve_owned(tmp_path, model)
    assert result.numerical_optimal and model.getVal(x) == 7
    model.freeProb()


def test_expired_owned_solver_keeps_only_existing_checked_incumbent(tmp_path):
    _, model = ScipAdapter.create_model('owned-timeout')
    x = model.addVar('x', vtype='INTEGER', lb=0)
    model.addCons(x >= 3)
    model.setObjective(x)
    assert ScipAdapter.warm_start(model, [(x, 8)], complete=True)
    _, result = solve_owned(tmp_path, model, seconds=.00001)
    assert result.status == 'time_limit' and result.primal == 8
    assert result.dual_bound is None and not result.numerical_optimal
    model.freeProb()
