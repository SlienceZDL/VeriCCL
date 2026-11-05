"""Native SCIP operations, numerical evidence and bounded lexicographic solves."""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import math
import threading
import time

from vericcl.errors import SemanticError, SolverUnavailableError
from vericcl.execution.budget import StandaloneLease


@dataclass(frozen=True)
class ScipSolveResult:
    status: str
    solver_status: str
    primal: float | None
    dual_bound: float | None
    relative_gap: float | None
    objective_values: tuple[float, ...]
    solution: dict[str, float]
    elapsed_s: float
    numerical_optimal: bool
    lexicographic_complete: bool
    evidence_scope: str = 'numerical_model'


class ScipModel:
    """A native model plus VeriCCL's primary-objective and budget bookkeeping."""
    def __init__(self, engine):
        self.engine = engine
        self.objectives = ()
        self.result = None
        self.lease = None
        self.configuration = {}
        self.indicator_slacks = []
        self._closed = False

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def addVar(self, *args, **kwargs):
        if self.lease is not None:
            self.lease.check()
        return self.engine.addVar(*args, **kwargs)

    def addCons(self, *args, **kwargs):
        if self.lease is not None:
            self.lease.check()
        return self.engine.addCons(*args, **kwargs)

    def general_constraint_count(self):
        return sum(constraint.getConshdlrName() != 'linear' for constraint in self.getConss())

    def getNSols(self):
        return self.engine.getNSols() if self.result is None else int(self.result.primal is not None)

    def getVal(self, expression):
        if self.result is not None and hasattr(expression, 'name') and expression.name in self.result.solution:
            return self.result.solution[expression.name]
        if self.result is not None and hasattr(expression, 'terms'):
            value = 0.
            for term, coefficient in expression.terms.items():
                product = float(coefficient)
                for variable in term:
                    product *= self.result.solution[variable.name]
                value += product
            return value
        return self.engine.getVal(expression)

    def setObjective(self, expression, sense='minimize', **kwargs):
        if sense != 'minimize':
            raise SemanticError('VeriCCL scheduling objectives must minimize')
        self.engine.setObjective(expression, sense, **kwargs)
        self.objectives = ((expression, 'primary'),)

    def set_lexicographic_objectives(self, objectives):
        objectives = tuple(objectives)
        if not objectives:
            raise SemanticError('at least one objective is required')
        self.engine.setObjective(objectives[0][0], 'minimize')
        self.objectives = objectives

    def freeProb(self):
        if not self._closed:
            self.engine.freeProb()
            self._closed = True

    def optimize(self):
        if self.lease is None:
            raise SemanticError('a SCIP model must have an explicit budget lease')
        return ScipAdapter.solve(self, self.lease)


class ScipAdapter:
    @staticmethod
    def available():
        return importlib.util.find_spec('pyscipopt') is not None

    @classmethod
    def require(cls):
        if not cls.available():
            raise SolverUnavailableError('SCIP/PySCIPOpt is unavailable; install the pinned solver dependency')
        try:
            return importlib.import_module('pyscipopt')
        except (ImportError, OSError) as error:
            raise SolverUnavailableError('SCIP/PySCIPOpt could not be loaded: ' + str(error)) from error

    @classmethod
    def create_model(cls, name, environment=None):
        if environment is not None:
            raise SemanticError('SCIP models own their native state; external solver environments are unsupported')
        module = cls.require()
        try:
            engine = module.Model(name)
            engine.hideOutput()
            return module, ScipModel(engine)
        except Exception as error:
            raise SolverUnavailableError('SCIP model creation failed: ' + str(error)) from error

    @classmethod
    def version(cls, module=None):
        _, model = cls.create_model('vericcl-version')
        try:
            return '.'.join(str(value) for value in (model.getMajorVersion(), model.getMinorVersion(), model.getTechVersion()))
        finally:
            model.freeProb()

    @staticmethod
    def configure(model, *, seed=0, threads=1, gap=0, lease=None):
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise SemanticError('SCIP thread allocation must be positive')
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1:
            raise SemanticError('SCIP seed must fit a nonnegative signed 32-bit integer')
        if not math.isfinite(gap) or not 0 <= gap <= 1:
            raise SemanticError('SCIP relative gap must be in [0, 1]')
        parameters = {'parallel/maxnthreads': threads, 'lp/threads': threads,
                      'randomization/randomseedshift': seed, 'limits/gap': gap, 'limits/absgap': 0.0}
        try:
            for key, value in parameters.items():
                model.setParam(key, value)
        except Exception as error:
            raise SolverUnavailableError('required SCIP parameter is unsupported: ' + str(error)) from error
        model.lease = lease
        model.configuration = {'allocated_threads': threads, 'search_threads': 1,
                               'lp_thread_limit': threads, 'parameters': parameters}
        return dict(model.configuration)

    @staticmethod
    def indicator(model, binary, active, constraint, *, name=''):
        # SCIP indicators accept one-sided inequalities. Equality is a pair of
        # native indicator constraints, never an arbitrary big-M relaxation.
        lhs, rhs, expression = constraint._lhs, constraint._rhs, constraint.expr
        results = []
        name = name or 'indicator-%d' % len(model.indicator_slacks)
        if lhs is not None:
            created = model.addConsIndicator(expression >= lhs, binvar=binary, activeone=bool(active), name=name + '-lb')
            results.append(created)
            model.indicator_slacks.append((model.getSlackVarIndicator(created), lhs - expression))
        if rhs is not None:
            created = model.addConsIndicator(expression <= rhs, binvar=binary, activeone=bool(active), name=name + '-ub')
            results.append(created)
            model.indicator_slacks.append((model.getSlackVarIndicator(created), expression - rhs))
        if not results:
            raise SemanticError('indicator requires a bounded linear constraint')
        return tuple(results)

    @staticmethod
    def warm_start(model, values, *, complete=False):
        solution = model.createSol() if complete else model.createPartialSol()
        for variable, value in values:
            if not math.isfinite(value):
                raise SemanticError('warm start values must be finite')
            model.setSolVal(solution, variable, value)
        if complete:
            # Native indicators introduce nonnegative slack variables. A
            # complete seed must set them even when its indicator is inactive.
            # The indicator handler still rejects positive slack when active.
            for slack, violation in model.indicator_slacks:
                model.setSolVal(solution, slack, max(0., float(model.getSolVal(solution, violation))))
        if complete and not model.checkSol(solution, printreason=False, completely=True):
            model.freeSol(solution)
            return False
        return bool(model.addSol(solution))

    @staticmethod
    def _snapshot(model, objectives):
        if model.getNSols() <= 0:
            return {}, ()
        solution = model.getBestSol()
        return ({var.name: float(model.getSolVal(solution, var)) for var in model.getVars(transformed=False)},
                tuple(float(model.getSolVal(solution, expression)) for expression, _ in objectives))

    @classmethod
    def solve(cls, model, lease):
        if not isinstance(model, ScipModel):
            raise SemanticError('SCIP solve requires a managed native model')
        cls.require()
        from vericcl.solver.worker_scope import current_solver_scope, solve_owned
        scope = current_solver_scope()
        if scope is not None:
            return solve_owned(model, lease, scope)
        model.result = None
        started = time.monotonic()
        objectives = model.objectives or ((model.getObjective(), 'primary'),)
        primary_bound, primary_gap, primary_optimal = None, None, False
        status, native_status, completed = 'not_run', 'unknown', 0
        solution, values = cls._snapshot(model, objectives)
        for index, (objective, name) in enumerate(objectives):
            if lease.cancelled or lease.remaining_seconds() <= 0:
                status = 'cancelled' if lease.cancelled else 'time_limit'
                break
            if index:
                previous = objectives[index - 1][0]
                optimum = values[index - 1]
                model.freeTransform()
                model.addCons(previous <= optimum, name='lex-fix-' + str(index - 1))
                model.engine.setObjective(objective, 'minimize')
                cls.warm_start(model, [(variable, solution[variable.name]) for variable in model.getVars(transformed=False)], complete=True)
            # SCIP's solving clock can include earlier phases. Set its limit to
            # elapsed solver time plus the remaining parent/model wall lease.
            model.setRealParam('limits/time', model.getSolvingTime() + lease.remaining_seconds())
            stopped = threading.Event()
            def interrupt_on_cancel():
                while not stopped.wait(.02):
                    if lease.cancelled or lease.remaining_seconds() <= 0:
                        model.engine.interruptSolve()
                        return
            watcher = threading.Thread(target=interrupt_on_cancel, daemon=True)
            watcher.start()
            try:
                model.engine.optimize()
            finally:
                stopped.set()
                watcher.join()
            native_status = str(model.getStatus())
            current_solution, current_values = cls._snapshot(model, objectives)
            if current_values:
                solution, values = current_solution, current_values
            if index == 0:
                bound = float(model.getDualbound())
                gap = float(model.getGap())
                primary_bound = bound if math.isfinite(bound) and abs(bound) < model.infinity() else None
                primary_gap = gap if math.isfinite(gap) and gap < model.infinity() else None
                primary_optimal = native_status == 'optimal'
            if lease.cancelled:
                status = 'cancelled'
                break
            if native_status == 'optimal':
                completed += 1
                status = 'optimal'
                continue
            if native_status == 'infeasible':
                status = 'infeasible' if index == 0 else 'error'
            elif native_status in {'timelimit', 'userinterrupt'} or lease.remaining_seconds() <= 0:
                status = 'time_limit'
            elif native_status in {'unbounded', 'inforunbd'}:
                status = 'unbounded'
            else:
                status = 'feasible' if values else 'not_run'
            break
        result = ScipSolveResult(status, native_status, values[0] if values else None, primary_bound,
                                 primary_gap, values, solution, time.monotonic() - started,
                                 primary_optimal, completed == len(objectives))
        model.result = result
        return result

    @classmethod
    def optimize_with_progress(cls, model, progress=None):
        result = model.optimize()
        if progress is not None:
            progress.best_value = math.inf if result.primal is None else result.primal
            progress.best_bound = 0. if result.dual_bound is None else result.dual_bound
            progress.mip_gap = math.inf if result.relative_gap is None else result.relative_gap
            progress.finished = result.numerical_optimal
        return result
