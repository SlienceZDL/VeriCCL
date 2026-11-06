from dataclasses import replace
import json
import time

import pytest

from tests.integration.test_lowering_v5 import candidate_fixture
from vericcl.execution.context import RunContext
from vericcl.xml.lower import lower_and_check


def test_zero_round_budget_and_model_work_restore_do_not_refund():
    from vericcl.execution.budget import BudgetManager, BudgetExhausted
    config = dict(stage_limits={'initial': (10, 10)}, cpu_budget=1, available_cpus=1,
                  per_model_timeout_s=4, cleanup_reserve_s=.1, max_tuning_rounds=0)
    budget = BudgetManager(**config)
    budget.restore_usage({'initial': 2}, started_rounds=0, elapsed_wall_s=3, model_work_s={'initial': 1})
    assert budget.snapshot()['model_work_s'] == {'initial': 1}
    with pytest.raises(BudgetExhausted):
        budget.start_tuning_round()


def test_checkpoint_restores_task_event_sequence_budget_and_complete_artifacts(tmp_path):
    from vericcl.workflow_state import save_checkpoint, restore_checkpoint, build_budgets
    candidate, lowering = candidate_fixture(tmp_path, 'broadcast')
    bundle = lower_and_check(candidate, lowering)
    run = RunContext.create(tmp_path, 'resume', 'r')
    budgets = build_budgets(lowering.inputs)
    budgets.restore_usage({'initial': 1}, started_rounds=1, elapsed_wall_s=2, model_work_s={'initial': .4})
    saved = save_checkpoint(run, lowering.inputs, budgets, lowering.model, bundle,
                            phase='tuning', search={'expanded': 2})
    resumed, state, recovered = restore_checkpoint(run.root)
    assert resumed.task_id == run.task_id and resumed.sequence > run.sequence
    assert state['search']['expanded'] == 2
    assert recovered.started_rounds == 1 and recovered.consumed['initial'] == 1
    assert recovered.model_work_s['initial'] == .4
    assert recovered.snapshot()['elapsed_wall_s'] >= 2
    assert state['bundle']['xml_sha256'] == bundle.xml_sha256
    (run.root / state['bundle']['xml']).write_text('damaged')
    with pytest.raises(ValueError, match='checksum'):
        restore_checkpoint(run.root)


def test_legacy_layout_checkpoint_retains_identity_and_budget_on_restore(tmp_path):
    from vericcl.workflow_state import save_checkpoint, restore_checkpoint, build_budgets
    from vericcl.execution.security import Redactor
    candidate, lowering = candidate_fixture(tmp_path, 'broadcast')
    bundle = lower_and_check(candidate, lowering)
    current = RunContext.create(tmp_path, 'legacy', 'r')
    for child in (current.root / 'process').iterdir():
        child.rename(current.root / child.name)
    (current.root / 'process').rmdir()
    metadata = json.loads((current.root / 'run.json').read_text())
    metadata['schema_version'] = 1
    (current.root / 'run.json').write_text(json.dumps(metadata))
    legacy = RunContext(current.root, task_id=current.task_id, run_id=current.run_id, redactor=Redactor())
    legacy.sequence = current.sequence
    budgets = build_budgets(lowering.inputs)
    budgets.restore_usage({'initial': 1}, started_rounds=1, elapsed_wall_s=2)
    save_checkpoint(legacy, lowering.inputs, budgets, lowering.model, bundle, phase='tuning')
    restored, state, budget = restore_checkpoint(legacy.root)
    assert restored.layout_version == 1
    assert not state['bundle']['xml'].startswith('process/')
    assert budget.started_rounds == 1 and budget.consumed['initial'] == 1
    assert restored.task_id == legacy.task_id


def test_public_resume_preserves_original_identity_during_disabled_live_migration(tmp_path, monkeypatch):
    from vericcl.cli.main import main
    from vericcl.input.json_codec import sha256_json
    from vericcl.input.schema import _freeze
    from vericcl.workflow_state import save_checkpoint, build_budgets, plain
    from vericcl.verification.online.model_publication import model_json
    from vericcl.workflow_v5 import PrimaryWorkflow
    seen = []
    original_init = PrimaryWorkflow.__init__
    def capture(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        seen.append(self.original.input_sha256)
    monkeypatch.setattr(PrimaryWorkflow, '__init__', capture)
    candidate, lowering = candidate_fixture(tmp_path, 'broadcast')
    config = plain(lowering.inputs.request_config)
    config['sketch']['live'] = {'enabled': False}
    config['sketch']['budgets']['max_tuning_rounds'] = 0
    identity = sha256_json(config)
    inputs = replace(lowering.inputs, input_sha256=identity, request_config=_freeze(config))
    bundle = lower_and_check(candidate, replace(lowering, inputs=inputs))
    assert bundle.can_execute
    run = RunContext.create(tmp_path, 'migrated', 'r')
    run.write_json('models/' + lowering.model.version + '.json', model_json(lowering.model))
    save_checkpoint(run, inputs, build_budgets(inputs), lowering.model, bundle, phase='tuning',
        extra={'original_config': config, 'purpose': 'solve', 'tune': False,
               'initial_model_version': lowering.model.version})
    assert main(['resume', '--run', str(run.root)]) == 0
    assert seen == [identity]
    latest = json.loads(run.artifact('checkpoints/latest.json').read_text())
    state = json.loads(run.artifact(latest['state']).read_text())
    assert state['input_sha256'] == identity
    assert 'live' not in state['extra']['original_config']['sketch']
    assert state['bundle']['input_sha256'] == identity


def test_history_matches_relative_bandwidth_and_adapts_integer_partition(tmp_path):
    from vericcl.solver.history_index import save_schedule, find_schedules, adapt_schedule
    from vericcl.planner.parameters import initial_parameter_domain, resolve_parameters
    candidate, lowering = candidate_fixture(tmp_path, 'broadcast')
    bundle = lower_and_check(candidate, lowering)
    path = tmp_path / 'history.json'
    save_schedule(path, bundle, lowering.inputs)
    hits = find_schedules(path, lowering.inputs, tolerance=.05, maximum=2, lease=lowering.lease)
    assert len(hits) == 1
    choice = next(iter(initial_parameter_domain(lowering.inputs, fixed_slice_bytes=2048).candidates))
    inputs = resolve_parameters(lowering.inputs, choice)
    adapted = adapt_schedule(hits[0], inputs)
    result = lower_and_check(adapted, replace(lowering, inputs=inputs))
    assert result.can_execute and result.program.chunks_per_loop == inputs.hyperparameters.slice_count
    assert 'miss_reasons' not in json.loads(path.read_text())


def test_history_rejects_changed_connectivity_and_bandwidth_ratios(tmp_path):
    from vericcl.solver.history_index import save_schedule, find_schedules
    from vericcl.input.schema import _freeze
    candidate, lowering = candidate_fixture(tmp_path, 'broadcast', ranks=3)
    bundle = lower_and_check(candidate, lowering)
    path = tmp_path / 'history.json'
    save_schedule(path, bundle, lowering.inputs)
    config = json.loads(json.dumps(lowering.inputs.request_config, default=dict))
    config['topology']['directed_links'][0]['beta_us_per_byte'] *= 4
    changed = replace(lowering.inputs, request_config=_freeze(config))
    assert not find_schedules(path, changed, tolerance=.1, maximum=2, lease=lowering.lease)
    config['topology']['directed_links'].pop()
    changed = replace(lowering.inputs, request_config=_freeze(config))
    assert not find_schedules(path, changed, tolerance=100, maximum=2, lease=lowering.lease)


@pytest.mark.parametrize('operator', ['broadcast', 'reduce', 'allgather', 'allreduce', 'alltoall', 'reduce_scatter'])
def test_history_integer_expansion_preserves_six_operator_semantics(tmp_path, operator):
    from vericcl.solver.history_index import save_schedule, find_schedules, adapt_schedule
    from vericcl.planner.parameters import initial_parameter_domain, resolve_parameters
    candidate, lowering = candidate_fixture(tmp_path, operator)
    bundle = lower_and_check(candidate, lowering)
    path = tmp_path / 'index.json'
    save_schedule(path, bundle, lowering.inputs)
    hit, = find_schedules(path, lowering.inputs, tolerance=0, maximum=1, lease=lowering.lease)
    choice = next(iter(initial_parameter_domain(lowering.inputs, fixed_slice_bytes=512).candidates))
    inputs = resolve_parameters(lowering.inputs, choice)
    result = lower_and_check(adapt_schedule(hit, inputs), replace(lowering, inputs=inputs))
    assert result.can_execute, {key: check.message for key, check in result.checks.items()}


def test_resume_reuses_prebaseline_candidates_without_restarting_synthesis(tmp_path, monkeypatch):
    from tests.integration.test_primary_lifecycle_v5 import request
    from vericcl.workflow_v5 import PrimaryWorkflow, run_primary
    from vericcl.workflow_state import restore_checkpoint
    from vericcl.verification.online.statistical_ledger import StatisticalLedger
    inputs, context = request(tmp_path)
    workflow = PrimaryWorkflow(inputs, context)
    context.write_json('models/' + workflow.model.version + '.json', __import__('vericcl.verification.online.model_publication', fromlist=['model_json']).model_json(workflow.model))
    workflow.initial_candidates()
    ledger = StatisticalLedger(context.artifact('statistics/attempts.jsonl'), task_id=context.task_id)
    first = ledger.reserve('fixture-A', 'fixture-B', seed=1)
    resumed, state, budgets = restore_checkpoint(context.root)
    initial_usage = budgets.consumed['initial']
    monkeypatch.setattr(PrimaryWorkflow, 'initial_candidates', lambda self: (_ for _ in ()).throw(AssertionError('must not restart synthesis')))
    result = run_primary(inputs, resumed, restored=state, budgets=budgets)
    assert result.contract_met and budgets.consumed['initial'] == initial_usage
    next_attempt = StatisticalLedger(resumed.artifact('statistics/attempts.jsonl'), task_id=resumed.task_id).reserve('fixture-C', 'fixture-D', seed=1)
    assert first.attempt_index == 1 and next_attempt.attempt_index == 2
