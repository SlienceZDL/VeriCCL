from dataclasses import replace
import json

import pytest

from tests.unit.online.test_trace_causality import trace_fixture, output_choices
from vericcl.execution.context import RunContext
from vericcl.input.schema import _freeze
from vericcl.verification.bottlenecks import AnalysisContext, SearchScope, analyze
from vericcl.verification.online.trace_causality import analyze_trace_causality
from vericcl.workflow_state import plain, restore_checkpoint
from vericcl.workflow_v5 import PrimaryRequest, PrimaryWorkflow, WorkflowServices


def test_observed_tail_changes_priority_with_same_model_and_preserves_search_space(tmp_path):
    bundle, lowering, _, _ = trace_fixture(tmp_path)
    left, right = output_choices(bundle)
    reports = []
    for ordinal, key in enumerate((left, right)):
        _, _, trace, _ = trace_fixture(tmp_path / str(ordinal), bundle=bundle, finish_us={key: 1000})
        reports.append(analyze_trace_causality(bundle, trace))
    scope = SearchScope('trace-priority', families=('path',), batch_size=1)
    results, causes, predictions = [], [], []
    for report in reports + [None]:
        context = AnalysisContext(lowering, trace_analysis=report)
        choices = []
        for _ in range(200):
            batch = analyze(bundle, lowering.model, scope, context)
            assert batch.status in {'opportunities', 'candidates_exhausted'}
            choices.extend(batch.strategies)
            if batch.status == 'candidates_exhausted':
                break
        else:
            pytest.fail('finite path fixture did not exhaust')
        results.append(choices)
        causes.append(batch.causes)
        predictions.append(batch.evidence['t_pred_us'])
        assert not batch.coverage['proven_no_improvement']
    assert len(set(predictions)) == 1
    assert results[0][0].strategy_id != results[1][0].strategy_id
    assert {s.strategy_id for s in results[0]} == {s.strategy_id for s in results[1]} == {s.strategy_id for s in results[2]}
    assert results[0][0].cause_ids and results[1][0].cause_ids
    assert any(cause.evidence.get('capture_id') == reports[0]['capture_id'] for cause in causes[0])
    assert not any(cause.evidence.get('capture_id') for cause in causes[2])


class RecordedDriver:
    """Only a workflow contract fixture; never actual device evidence."""
    def __init__(self, raw):
        self.raw, self.calls = replace(raw, driver_kind='native_msccl'), 0
    def trace_driver(self, bundle, inputs, lease):
        return self
    def capture_once(self, plan):
        from vericcl.verification.online.trace_format import RAW_HEADER_STRUCT
        self.calls += 1
        paths = {}
        for rank, source in self.raw.paths.items():
            data = bytearray(source.read_bytes())
            header = list(RAW_HEADER_STRUCT.unpack_from(data))
            header[6] = plan.capacity_by_rank[rank]
            RAW_HEADER_STRUCT.pack_into(data, 0, *header)
            path = source.with_name(source.stem + '-captured.bin')
            path.write_bytes(data)
            paths[rank] = path
        return replace(self.raw, paths=paths)


def workflow_fixture(tmp_path, *, enabled=True, partial=False):
    bundle, lowering, _, raw = trace_fixture(tmp_path)
    if partial:
        paths = dict(raw.paths)
        paths.pop(2)
        raw = replace(raw, paths=paths)
    config = plain(lowering.inputs.request_config)
    config['sketch']['trace_and_measure'].update(enabled=enabled, job_timeout_s=2, trace_capacity_bytes=1048576)
    config['sketch']['budgets'].update(max_online_validation_per_round_s=10,
        max_online_validation_total_s=20, max_tuning_rounds=0)
    inputs = replace(lowering.inputs, request_config=_freeze(config))
    run = RunContext.create(tmp_path, 'trace-workflow', 'r')
    driver = RecordedDriver(raw)
    request = PrimaryRequest(inputs, purpose='verify', xml_text=bundle.xml_text, sidecar=bundle.sidecar,
                             cost_model=lowering.model)
    return PrimaryWorkflow(request, run, WorkflowServices(native_driver=driver)), driver


def test_workflow_capture_feeds_analysis_and_checkpoint_without_recapture(tmp_path, monkeypatch):
    from vericcl import workflow_v5
    workflow, driver = workflow_fixture(tmp_path)
    result = workflow.run()
    assert result.contract_met and driver.calls == 1
    reference = workflow.best_trace
    report = json.loads(workflow.context.artifact(reference['path']).read_text())
    assert report['xml_sha256'] == workflow.best.xml_sha256
    assert report['completion_tails']
    actual_analyze, received = workflow_v5.analyze, []
    def observe(bundle, model, scope, context):
        received.append(context.trace_analysis)
        return replace(actual_analyze(bundle, model, scope, context), strategies=())
    monkeypatch.setattr(workflow_v5, 'certify_global_cut', lambda *args: None)
    monkeypatch.setattr(workflow_v5, 'analyze', observe)
    workflow.tuning_round()
    assert received[0]['capture_id'] == report['capture_id']
    assert driver.calls == 1
    context, state, budgets = restore_checkpoint(workflow.context.root)
    restored = PrimaryWorkflow(workflow.request, context, WorkflowServices(native_driver=driver),
                               restored=state, budgets=budgets)
    assert restored.run().contract_met
    assert restored.best_trace == reference and driver.calls == 1
    assert not restored.adopt(restored.best, restored.best_inputs, restored.best_record,
        evidence={'measured_required': True, 'trace_complete': True, 'trace_evidence_kind': 'native_msccl',
                  'trace_causality': {'path': 'unrelated.json'}})
    assert restored.best_trace == reference
    workflow.context.artifact(reference['path']).write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        restore_checkpoint(workflow.context.root)


def test_partial_capture_persists_independent_region_but_keeps_acceptance_gate(tmp_path):
    workflow, driver = workflow_fixture(tmp_path, partial=True)
    result = workflow.run()
    assert not result.contract_met and result.stop_reason == 'no_qualified_baseline'
    paths = list(workflow.context.artifact('reports').glob('trace-causality-*.json'))
    assert len(paths) == 1 and driver.calls == 1
    report = json.loads(paths[0].read_text())
    assert report['status'] == 'partial'
    assert any(row['endpoint_complete'] for row in report['operations'].values())
    assert workflow.best_trace is None


def test_disabled_trace_preserves_default_workflow_and_never_captures(tmp_path):
    workflow, driver = workflow_fixture(tmp_path, enabled=False)
    assert workflow.run().contract_met
    assert driver.calls == 0 and workflow.best_trace is None
    assert not list(workflow.context.artifact('reports').glob('trace-causality-*.json'))


def test_analysis_rejects_trace_from_another_xml(tmp_path):
    bundle, lowering, trace, _ = trace_fixture(tmp_path)
    report = analyze_trace_causality(bundle, trace)
    report['xml_sha256'] = '0' * 64
    context = AnalysisContext(lowering, trace_analysis=report)
    batch = analyze(bundle, lowering.model, SearchScope('wrong-trace'), context)
    assert batch.status == 'analysis_error' and not batch.strategies


def test_legacy_measured_checkpoint_without_bound_report_requires_explicit_reverification(tmp_path):
    workflow, driver = workflow_fixture(tmp_path)
    assert workflow.run().contract_met
    context, state, budgets = restore_checkpoint(workflow.context.root)
    state['extra'].pop('best_trace')
    restored = PrimaryWorkflow(workflow.request, context, WorkflowServices(native_driver=driver),
                               restored=state, budgets=budgets)
    result = restored.run()
    assert not result.contract_met and result.stop_reason == 'execution_error'
    assert any('reverify its XML in a new run' in failure for failure in result.optional_failures)
    assert driver.calls == 1


def test_observed_joint_region_does_not_remove_smaller_model_domain():
    from types import SimpleNamespace
    from vericcl.execution.budget import StandaloneLease
    from vericcl.verification.bottlenecks import _default_source
    keys = ('a', 'b', 'c')
    bundle = SimpleNamespace(storage_plan=SimpleNamespace(instructions=tuple(
        SimpleNamespace(operation_id=key, kind='copy') for key in keys)))
    lowering = SimpleNamespace(topology=None, inputs=None, lease=StandaloneLease.after(10))
    index = SimpleNamespace(tails=('b',), predecessors={'a': (), 'b': ('a',), 'c': ()},
                            related={key: () for key in keys})
    original = {row.operation_ids for row in _default_source('joint', bundle, index, AnalysisContext(lowering))}
    report = dict(completion_tails=['c'], local_completion_tails={},
                  operations={key: {'overlap_groups': []} for key in keys}, resource_overlap_groups={})
    observed = {row.operation_ids for row in _default_source('joint', bundle, index,
        AnalysisContext(lowering, trace_analysis=report))}
    assert ('a', 'b') in original
    assert original <= observed


@pytest.mark.parametrize('family', ['order', 'channel', 'joint'])
def test_observation_preserves_all_existing_legal_family_decisions(tmp_path, family):
    from vericcl.execution.budget import StandaloneLease
    bundle, lowering, trace, _ = trace_fixture(tmp_path)
    scope = SearchScope('preserved-' + family, families=(family,), batch_size=16)
    choices = []
    for report in (None, analyze_trace_causality(bundle, trace)):
        context = AnalysisContext(replace(lowering, lease=StandaloneLease.after(30)), trace_analysis=report)
        selected = set()
        for _ in range(200):
            batch = analyze(bundle, lowering.model, scope, context)
            assert batch.status in {'opportunities', 'candidates_exhausted'}
            selected.update(row.strategy_id for row in batch.strategies)
            if batch.status == 'candidates_exhausted':
                break
        else:
            pytest.fail('finite strategy fixture did not exhaust')
        choices.append(selected)
    assert choices[0] <= choices[1]
