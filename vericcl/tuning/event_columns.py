"""Native SCIP disjunction of concrete route/channel/order event patterns.

Columns are expanded lazily by the repair search. Within each column the model
conserves bytes under the exact active-mode rates and enforces eager starts.
Selecting a column jointly selects its integer slice/path/channel/order choices.
A batch optimum is numerical evidence for that batch, not full-space proof.
"""
from fractions import Fraction

from vericcl.errors import SemanticError
from vericcl.performance.simulation import _components, _profile, _rates
from vericcl.solver.scip_api import ScipAdapter
from vericcl.solver.trees.operators import model_lease


def _groups(prediction):
    groups = {}
    kinds = {'start': 'start', 'startup_complete': 'active', 'complete': 'finish'}
    for event in prediction.events:
        groups.setdefault(event.time_us, []).append((event.operation_id, kinds[event.kind]))
    return tuple((stamp, tuple(sorted(values))) for stamp, values in sorted(groups.items()))


def solve_event_columns(bundles, context):
    bundles = tuple(bundles)
    if not bundles or any(not bundle.can_execute for bundle in bundles):
        raise SemanticError('event columns require fully checked executable candidates')
    model_data = context.lowering.model
    if any(bundle.prediction.model_version != model_data.version for bundle in bundles):
        raise SemanticError('event columns must share the current model')
    with model_lease(context.lowering.inputs, context) as (lease, threads):
        module, model = ScipAdapter.create_model('vericcl-local-event-columns')
        try:
            ScipAdapter.configure(model, seed=context.lowering.inputs.solver.solver_seed,
                                  threads=threads, gap=0, lease=lease)
            horizon = max(bundle.prediction.t_pred_us for bundle in bundles)
            objective = model.addVar('complete_time', lb=0, ub=horizon)
            choices = [model.addVar('column-%d' % index, vtype='B') for index in range(len(bundles))]
            model.addCons(module.quicksum(choices) == 1)
            selected_seed = min(range(len(bundles)), key=lambda i: (bundles[i].prediction.t_pred_us, bundles[i].xml_sha256))
            warm = [(objective, bundles[selected_seed].prediction.t_pred_us)]
            warm.extend((var, int(i == selected_seed)) for i, var in enumerate(choices))
            for index, bundle in enumerate(bundles):
                lease.check()
                chosen, selected = choices[index], index == selected_seed
                operations = {op.operation_id: op for op in bundle.executable.operations}
                costs = {key: model_data.service_for(op.cost_key, op.payload_bytes) for key, op in operations.items() if op.kind != 'nop'}
                groups = _groups(bundle.prediction)
                times = [model.addVar('column-%d-time-%d' % (index, i), lb=0, ub=horizon) for i in range(len(groups))]
                warm.extend((var, stamp if selected else 0.) for var, (stamp, _) in zip(times, groups))
                if not times:
                    continue
                model.addCons(times[-1] <= horizon * chosen)
                model.addCons(objective >= times[-1])
                for a, b in zip(times, times[1:]):
                    model.addCons(b >= a)
                transitions = {(key, kind): i for i, (_, group) in enumerate(groups) for key, kind in group}
                present, active, rates, startup = set(), set(), [], {}
                for i, (_, group) in enumerate(groups):
                    ending = {key for key, kind in group if kind == 'finish'}
                    starting = {key for key, kind in group if kind == 'start'}
                    activating = {key for key, kind in group if kind == 'active'}
                    present.difference_update(ending); active.difference_update(ending); present.update(starting)
                    for component in _components([operations[key] for key in sorted(present) if operations[key].kind in {'send', 'rrc'}]):
                        profile = _profile(component, model_data)
                        if profile:
                            for op in component:
                                if op.operation_id in starting and op.flow_id in profile.startup_us:
                                    startup[op.operation_id] = profile.startup_us[op.flow_id]
                    active.update(activating - ending)
                    vector, _ = _rates([operations[key] for key in sorted(active) if operations[key].kind != 'nop'],
                                       costs, model_data, set(), lambda v: Fraction(str(v)), set())
                    rates.append(vector)
                for ordinal, (key, op) in enumerate(operations.items()):
                    lease.check()
                    start, active_time, finish = (times[transitions[key, kind]] for kind in ('start', 'active', 'finish'))
                    release = op.not_before_us
                    model.addCons(start >= release * chosen)
                    if op.dependencies:
                        ready_sources = [(release, release * chosen)] + [(bundle.prediction.end_times[dep], times[transitions[dep, 'finish']]) for dep in op.dependencies]
                        witnesses = [model.addVar('column-%d-eager-%d-%d' % (index, ordinal, j), vtype='B') for j in range(len(ready_sources))]
                        model.addCons(module.quicksum(witnesses) == chosen)
                        maximum = max(range(len(ready_sources)), key=lambda j: ready_sources[j][0])
                        for j, (variable, (_, ready)) in enumerate(zip(witnesses, ready_sources)):
                            model.addCons(start >= ready)
                            ScipAdapter.indicator(model, variable, True, start == ready)
                            warm.append((variable, int(selected and j == maximum)))
                    else:
                        model.addCons(start == release * chosen)
                    if op.kind == 'nop':
                        model.addCons(active_time == start); model.addCons(finish == start)
                    else:
                        model.addCons(active_time - start == startup.get(key, costs[key].alpha_us) * chosen)
                        model.addCons(module.quicksum(float(rates[i].get(key, 0) / op.payload_bytes) * (times[i+1] - times[i])
                                                      for i in range(len(times)-1)) == chosen)
            model.setObjective(objective)
            if not ScipAdapter.warm_start(model, warm, complete=True):
                raise SemanticError('complete exact-event incumbent failed native model feasibility checks')
            context.model_calls += 1
            result = model.optimize()
            if result.primal is None:
                return None, result
            selected = max(range(len(choices)), key=lambda i: result.solution[choices[i].name])
            if result.solution[choices[selected].name] < .5:
                raise SemanticError('native event model did not select an integer column')
            actual = bundles[selected].prediction.t_pred_us
            if abs(actual - result.primal) > 1e-6 * max(1., actual):
                raise SemanticError('native event completion disagrees with independent physical replay')
            return selected, result
        finally:
            model.freeProb()
