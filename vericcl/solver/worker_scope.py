"""Fresh SCIP workers with serialized models, owned deadlines and checked results.

The worker entry point (``python -m vericcl.solver.worker_scope DIR``) imports
only the SCIP adapter and its budget/error helpers. Process ownership, run
records and artifact writers are parent-side concerns and are imported lazily,
so a fresh worker does not pay for loading the synthesis/verification stack.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

from vericcl.errors import SemanticError


_SCOPE = ContextVar('vericcl_owned_solver_scope', default=None)


@dataclass
class SolverScope:
    run_context: object
    cleanup_deadline: float
    owner_pid: int


@contextmanager
def isolated_solver_scope(run_context, *, cleanup_deadline):
    token = _SCOPE.set(SolverScope(run_context, cleanup_deadline, os.getpid()))
    try:
        yield
    finally:
        _SCOPE.reset(token)


def current_solver_scope():
    scope = _SCOPE.get()
    return scope if scope is not None and scope.owner_pid == os.getpid() else None


def _linear(expression):
    if isinstance(expression, (int, float)):
        return {'constant': float(expression), 'terms': {}}
    result = {'constant': 0., 'terms': {}}
    for term, coefficient in expression.terms.items():
        if len(term) > 1:
            raise SemanticError('owned SCIP objectives must be linear')
        if len(term) == 0:
            result['constant'] += float(coefficient)
        else:
            result['terms'][term[0].name] = float(coefficient)
    return result


def _checked_install(model, solution, objectives, objective_values):
    variables = {variable.name: variable for variable in model.engine.getVars(transformed=False)}
    if set(solution) != set(variables) or any(not math.isfinite(value) for value in solution.values()):
        raise SemanticError('owned SCIP result has an invalid variable assignment')
    candidate = model.engine.createSol()
    try:
        for name, value in solution.items():
            model.engine.setSolVal(candidate, variables[name], value)
        if not model.engine.checkSol(candidate, printreason=False, completely=True):
            raise SemanticError('owned SCIP result failed independent parent-model feasibility checking')
        values = tuple(float(model.engine.getSolVal(candidate, expression)) for expression, _ in objectives)
        if len(values) != len(objective_values) or any(not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-7)
                                                      for a, b in zip(values, objective_values)):
            raise SemanticError('owned SCIP result has inconsistent objective values')
        # A valid duplicate may be rejected by the solution pool. Feasibility
        # and objective checking above, rather than addSol's return, are the gate.
        model.engine.addSol(candidate, free=False)
    finally:
        model.engine.freeSol(candidate)


_RESULT_CACHE_LIMIT = 512


def _reusable(values):
    """Only proven outcomes are independent of the remaining time budget."""
    return ((values.get('status') == 'optimal' and values.get('lexicographic_complete') is True)
            or values.get('status') == 'infeasible')


def _result_cache(context):
    cache = getattr(context, 'solver_results', None)
    if cache is None:
        cache = {}
        context.solver_results = cache
    return cache


def _model_key(directory, request):
    digest = hashlib.sha256()
    for name in ('model.cip', 'model.set'):
        data = (directory / name).read_bytes()
        digest.update(len(data).to_bytes(8, 'little'))
        digest.update(data)
    stable = {key: request[key] for key in ('objectives', 'configuration', 'warm_solution')}
    digest.update(json.dumps(stable, sort_keys=True, allow_nan=False).encode())
    return digest.hexdigest()


def solve_owned(model, lease, scope):
    from vericcl.execution.processes import ProcessLease, run_owned_process
    from vericcl.solver.scip_api import ScipAdapter, ScipSolveResult
    model.result = None
    started = time.monotonic()
    objectives = model.objectives or ((model.getObjective(), 'primary'),)
    warm_solution, warm_values = ScipAdapter._snapshot(model, objectives)

    def interrupted(status):
        outcome = ScipSolveResult(status, 'owned_worker_' + status,
            warm_values[0] if warm_values else None, None, None, warm_values, warm_solution,
            time.monotonic() - started, False, False)
        model.result = outcome
        return outcome

    if lease.cancelled or lease.remaining_seconds() <= 0:
        return interrupted('cancelled' if lease.cancelled else 'time_limit')
    hard = min(scope.cleanup_deadline, getattr(lease, 'hard_deadline', scope.cleanup_deadline))
    work = min(lease.work_deadline, hard)
    if hard <= work:
        raise SemanticError('owned solver needs a positive cleanup reservation within its parent deadline')
    context = scope.run_context
    identifier = 'scip-' + uuid.uuid4().hex
    directory = context.artifact('tmp/' + identifier)
    directory.mkdir()
    variables = model.engine.getVars(transformed=False)
    if len({item.name for item in variables}) != len(variables):
        raise SemanticError('owned SCIP model variable names must be unique')
    model.engine.writeProblem(str(directory / 'model.cip'), trans=False, genericnames=False, verbose=False)
    model.engine.writeParams(str(directory / 'model.set'), comments=False, onlychanged=True, verbose=False)
    request = {
        'objectives': [{'name': name, **_linear(expression)} for expression, name in objectives],
        'configuration': model.configuration, 'warm_solution': warm_solution,
        'work_deadline': work, 'parent_pid': os.getpid(),
    }
    context.write_scratch_json('tmp/' + identifier + '/request.json', request)
    key = _model_key(directory, request)
    cache = _result_cache(context)
    reused = cache.get(key)
    if reused is not None:
        # A byte-identical model, parameter file, objective list, warm start
        # and thread allocation already finished with a proof (optimal or
        # infeasible) in this run. SCIP is deterministic for identical input,
        # so a fresh worker would return the same outcome; the parent still
        # rechecks feasibility and objective values independently below.
        source, values = reused
        outcome = ScipSolveResult(**{**values, 'objective_values': tuple(values['objective_values']),
                                     'solution': dict(values['solution']), 'elapsed_s': time.monotonic() - started})
        if outcome.primal is not None:
            _checked_install(model, outcome.solution, objectives, outcome.objective_values)
        context.write_scratch_json('tmp/' + identifier + '/reused.json',
                                   {'source_job': source, 'model_sha256': key, 'status': outcome.status})
        from vericcl.execution.events import emit_event
        emit_event(context, 'solver_result_reused', {'job_id': identifier, 'source_job': source,
                                                     'model_sha256': key, 'status': outcome.status})
        model.result = outcome
        return outcome
    result = run_owned_process(context, (sys.executable, '-m', 'vericcl.solver.worker_scope', str(directory)),
        ProcessLease(hard, hard - work), cwd=Path(__file__).resolve().parents[2], job_id=identifier,
        cancel_check=lambda: lease.cancelled, durable_records=False)
    if not result.cleanup_complete:
        lease.ownership_unresolved = True
        raise SemanticError('owned SCIP process did not confirm cleanup; CPU resources remain occupied')
    output = directory / 'result.json'
    if result.status == 'completed' and output.is_file():
        values = json.loads(output.read_text())
        values['objective_values'] = tuple(values['objective_values'])
        outcome = ScipSolveResult(**values)
        if outcome.primal is not None:
            _checked_install(model, outcome.solution, objectives, outcome.objective_values)
        if _reusable(values):
            if len(cache) >= _RESULT_CACHE_LIMIT:
                cache.pop(next(iter(cache)))
            cache[key] = (identifier, values)
        model.result = outcome
        return outcome
    if result.status in {'timeout', 'not_started', 'cancelled'}:
        return interrupted('cancelled' if result.status == 'cancelled' else 'time_limit')
    raise SemanticError('native SCIP worker failed; see jobs/' + identifier)


def _publish_scratch(path, data):
    """Atomic no-replace publication of a worker scratch result.

    The parent reads ``result.json`` only after this process has exited, and
    ``tmp/`` is rebuildable task scratch, so durability (fsync) is not needed;
    the temporary-name plus link publication still never exposes a partial
    result and never replaces an existing one.
    """
    path = Path(path)
    temporary = path.with_name('.{}.{}.tmp'.format(path.name, uuid.uuid4().hex))
    try:
        with temporary.open('xb') as stream:
            stream.write(data)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _worker(directory):
    from vericcl.execution.budget import StandaloneLease
    from vericcl.solver.scip_api import ScipAdapter
    request = json.loads((directory / 'request.json').read_text())
    # Linux parent-death protection complements the owner's bounded group kill.
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, 9, 0, 0, 0) != 0 or os.getppid() != request['parent_pid']:
        raise RuntimeError('SCIP worker parent ownership was lost')
    _, model = ScipAdapter.create_model('vericcl-owned-worker')
    try:
        model.engine.readProblem(str(directory / 'model.cip'))
        variables = {variable.name: variable for variable in model.engine.getVars(transformed=False)}
        objectives = tuple((item['constant'] + sum(value * variables[name] for name, value in item['terms'].items()),
                            item['name']) for item in request['objectives'])
        model.set_lexicographic_objectives(objectives)
        lease = StandaloneLease(request['work_deadline'])
        configuration = request['configuration']
        parameters = configuration.get('parameters', {})
        ScipAdapter.configure(model, seed=parameters.get('randomization/randomseedshift', 0),
            threads=configuration.get('allocated_threads', 1), gap=parameters.get('limits/gap', 0), lease=lease)
        model.engine.readParams(str(directory / 'model.set'))
        warm = request['warm_solution']
        if warm:
            ScipAdapter.warm_start(model, [(variables[name], value) for name, value in warm.items()], complete=True)
        outcome = ScipAdapter.solve(model, lease)
        _publish_scratch(directory / 'result.json', (json.dumps(asdict(outcome), allow_nan=False) + '\n').encode())
    finally:
        model.freeProb()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: python -m vericcl.solver.worker_scope OWNED_JOB_DIRECTORY')
    _worker(Path(sys.argv[1]).resolve())
