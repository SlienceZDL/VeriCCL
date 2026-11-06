"""Reproduce local repair across slices, history adaptation, and beneficial resynthesis with a new slice size S using a tutorial model."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

from tools.generate_examples import example_documents, write_documents
from vericcl.execution.budget import StandaloneLease
from vericcl.execution.context import RunContext, finalize_run
from vericcl.execution.status import RunResult
from vericcl.input.loader import load_request
from vericcl.performance.initial import initial_cost_model
from vericcl.planner.parameters import initial_parameter_domain, resolve_parameters
from vericcl.solver.history_index import save_schedule, find_schedules, adapt_schedule
from vericcl.topology import load_topology
from vericcl.tuning.strategies import RepairContext, build_domain, solve_repair, outside_invariants
from vericcl.tuning.resynthesis import ResynthesisRequest, resynthesize_branch
from vericcl.verification.bottlenecks import StrategyRecord
from vericcl.verification.online.model_publication import model_json
from vericcl.workflow_state import plain
from vericcl.xml.lower import LoweringContext, lower_and_check
from vericcl.xml.v5_model import NetworkCandidate, NetworkOperation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--case', default='repair-tutorial')
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args(argv)
    context = RunContext.create(args.project_root, args.case, args.run_id)
    documents = example_documents('broadcast', ranks=3)
    documents['sketch']['hyperparameters']['total_size_bytes'] = 4096
    documents['atom']['stage_num'] = 1
    for link in documents['topology']['directed_links']:
        link['beta_us_per_byte'] = .1 if (link['src'], link['dst']) == (0, 1) else .01
    paths = write_documents(context.root.parent.parent / 'input' / args.run_id, documents)
    for key, path in paths.items(): context.write_bytes('input_snapshot/' + key + '.json', path.read_bytes())
    context.write_json('input-snapshot.json', {'files': {
        key + '.json': hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}}, replace=True)
    inputs = load_request(paths); model = initial_cost_model(inputs)
    context.write_json('effective-request.json', plain(inputs.request_config))
    context.write_json('value-sources.json', plain(inputs.value_sources))
    context.write_json('invocation.json', {'argv': [sys.executable, '-m', 'tools.repair_example', *(argv or sys.argv[1:])]})
    context.write_json('models/initial.json', model_json(model))
    lowering = LoweringContext(inputs, load_topology(inputs), model, StandaloneLease.after(60), run_context=context)
    operations = tuple(NetworkOperation('slice%d-to%d' % (s, target), 'send', 0, target, 0, 0,
        frozenset({s}), frozenset({s}), (0, 1, 2), frozenset(), frozenset(), float(s))
        for s in range(4) for target in (1, 2))
    baseline = lower_and_check(NetworkCandidate(3, 4, 1024, operations, 'explicit_teaching_schedule'), lowering)
    ids = ('slice0-to1', 'slice1-to1')
    strategy = StrategyRecord('allocation', ids, paths=tuple((key, (0, 2, 1)) for key in ids))
    domain = build_domain(None, strategy); repair = RepairContext(lowering)
    result = solve_repair(baseline, domain, strategy, repair)
    if result.status != 'improved': raise RuntimeError('expected local improvement: ' + result.status)
    outside = outside_invariants(baseline, result.candidate, domain)
    if not outside.valid: raise RuntimeError('outside decisions changed')
    index = context.artifact('models/history.json')
    save_schedule(index, result.candidate, inputs)
    hit, = find_schedules(index, inputs, tolerance=0, maximum=1, lease=lowering.lease)
    choice = next(iter(initial_parameter_domain(inputs, fixed_slice_bytes=2048).candidates))
    changed = resolve_parameters(inputs, choice)
    repartitioned = lower_and_check(adapt_schedule(hit, changed), replace(lowering, inputs=changed))
    if not repartitioned.can_execute or repartitioned.prediction.t_pred_us >= result.candidate.prediction.t_pred_us:
        raise RuntimeError('the example needs an independently checked partition benefit')
    branch = resynthesize_branch(result.candidate, ResynthesisRequest('partition', slice_bytes=2048,
        reason='validated complete-slice adaptation improves current-model completion'), repair)
    try: resynthesized = next(iter(branch))
    finally: branch.close()
    bundles = {'before': baseline, 'local-repair': result.candidate, 'adapted-history': repartitioned,
               'resynthesized': resynthesized.candidate}
    for name, bundle in bundles.items():
        if not bundle.can_execute: raise RuntimeError('example candidate failed final checks: ' + name)
        context.write_bytes('candidates/' + name + '.xml', bundle.xml_text.encode())
        context.write_json('candidates/' + name + '.sidecar.json', bundle.sidecar)
    report = {'scope': 'CPU/SCIP with explicit teaching costs; no GPU execution or global optimum claim',
        'domain': list(ids), 'outside_valid': outside.valid,
        'completion_us': {name: bundle.prediction.t_pred_us for name, bundle in bundles.items()},
        'old_slice_bytes': 1024, 'new_slice_bytes': 2048, 'history_hits': 1,
        'parent_lease_preserved': resynthesized.context.lease is lowering.lease,
        'outer_round_reset': branch.outer_round_reset, 'native_local_models': result.model_calls}
    context.write_json('reports/example.json', report)
    context.write_json('reports/solver-evidence.json', list(result.solver_evidence))
    context.write_json('reports/repair-attempts.json', list(result.attempts))
    finalized = finalize_run(context, RunResult(True, 'completed', stop_reason='teaching_example'))
    print(json.dumps(report, ensure_ascii=False))
    return finalized.exit_code


if __name__ == '__main__':
    raise SystemExit(main())
