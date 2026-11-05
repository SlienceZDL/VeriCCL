"""Optimize event intervals for a specified order using the shared service model.

Discrete order search belongs to the caller. This model optimizes actual byte
progress within that order, then replays the result through the authoritative
predictor. Its numerical optimum does not certify other event orders.
"""
from dataclasses import dataclass, replace
from fractions import Fraction

from vericcl.errors import SemanticError
from vericcl.performance.events import Executable
from vericcl.performance.simulation import _components, _profile, _rates, predict
from vericcl.solver.scip_api import ScipAdapter


@dataclass(frozen=True)
class EventStep:
    kind: str
    operation_id: str


@dataclass(frozen=True)
class EventOrderResult:
    solver_result: object
    executable: object = None
    prediction: object = None
    scope: str = 'fixed_event_order_numerical'
    full_search_proven: bool = False
    model_replay_consistent: bool = False


def optimize_event_order(executable, cost_model, order, lease, *, threads=1, seed=0):
    if dict(executable.context) != dict(cost_model.context):
        raise SemanticError('event model and executable contexts differ')
    operations = {op.operation_id: op for op in executable.operations}
    order = tuple(tuple(group) for group in order)
    transitions = {}
    for index, group in enumerate(order):
        if not group:
            raise SemanticError('event transition group cannot be empty')
        for event in group:
            key = event.operation_id, event.kind
            if event.operation_id not in operations or event.kind not in {'start', 'active', 'finish'} or key in transitions:
                raise SemanticError('invalid or duplicate operation transition')
            transitions[key] = index
    required = {(key, kind) for key in operations for kind in ('start', 'active', 'finish')}
    if set(transitions) != required or any(not (transitions[key, 'start'] <= transitions[key, 'active'] <= transitions[key, 'finish']) for key in operations):
        raise SemanticError('event order must cover every operation transition in order')
    costs = {}
    for key, op in operations.items():
        if op.kind != 'nop':
            costs[key] = cost_model.service_for(op.cost_key, op.payload_bytes)
            if costs[key] is None:
                raise SemanticError('missing event service cost: ' + op.cost_key)
        if set(op.resources) - cost_model.resources.keys():
            raise SemanticError('missing event resource capacity')
    module, model = ScipAdapter.create_model('vericcl-event-intervals')
    ScipAdapter.configure(model, seed=seed, threads=threads, gap=0, lease=lease)
    try:
        times = [model.addVar(name='event-' + str(i), lb=0) for i in range(len(order))]
        for left, right in zip(times, times[1:]):
            model.addCons(right >= left)
        present, active = set(), set()
        interval_rates = []
        startup = {}
        for index, group in enumerate(order):
            lease.check()
            finishing = {event.operation_id for event in group if event.kind == 'finish'}
            starting = {event.operation_id for event in group if event.kind == 'start'}
            activating = {event.operation_id for event in group if event.kind == 'active'}
            present.difference_update(finishing)
            active.difference_update(finishing)
            present.update(starting)
            for subgroup in _components([operations[key] for key in sorted(present) if operations[key].kind in {'send', 'rrc'}]):
                profile = _profile(subgroup, cost_model)
                if profile is not None:
                    for op in subgroup:
                        if op.operation_id in starting and op.flow_id in profile.startup_us:
                            startup[op.operation_id] = profile.startup_us[op.flow_id]
            active.update(activating - finishing)
            rates, _ = _rates([operations[key] for key in sorted(active) if operations[key].kind != 'nop'],
                              costs, cost_model, set(), lambda value: Fraction(str(value)), set())
            interval_rates.append(rates)
        for key, op in operations.items():
            start, ready, finish = (times[transitions[key, kind]] for kind in ('start', 'active', 'finish'))
            model.addCons(start >= op.not_before_us)
            for predecessor in op.dependencies:
                model.addCons(start >= times[transitions[predecessor, 'finish']])
            if op.kind == 'nop':
                model.addCons(ready == start)
                model.addCons(finish == start)
                continue
            model.addCons(ready - start == startup.get(key, costs[key].alpha_us))
            # Normalize to one payload to avoid coefficients growing with M.
            model.addCons(module.quicksum(float(interval_rates[i].get(key, 0) / op.payload_bytes) * (times[i + 1] - times[i])
                                          for i in range(len(times) - 1)) == 1)
        model.setObjective(times[-1] if times else module.quicksum(()), 'minimize')
        result = ScipAdapter.solve(model, lease)
        if result.primal is None:
            return EventOrderResult(result)
        candidate = Executable(tuple(replace(op, not_before_us=max(op.not_before_us,
                                     model.getVal(times[transitions[op.operation_id, 'start']]))) for op in executable.operations),
                               executable.output_operations, executable.context)
        prediction = predict(candidate, cost_model, exact=True)
        consistent = prediction.t_pred_us is not None and abs(prediction.t_pred_us - result.primal) <= 1e-6 * max(1, abs(result.primal))
        return EventOrderResult(result, candidate, prediction, model_replay_consistent=consistent)
    finally:
        model.freeProb()
