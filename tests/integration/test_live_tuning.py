from dataclasses import replace

from tests.integration.test_lowering_v5 import candidate_fixture
from vericcl.execution.budget import StandaloneLease
from vericcl.execution.context import RunContext
from vericcl.live.triggers import TriggerEvent
from vericcl.xml.lower import lower_and_check


def test_explicit_failed_edge_is_removed_and_locally_rerouted_without_outside_changes(tmp_path):
    from vericcl.live.tuning import repair_failed_links, restricted_inputs, used_links
    from vericcl.topology import load_topology
    (tmp_path / 'inputs').mkdir()
    candidate, context = candidate_fixture(tmp_path / 'inputs', 'broadcast', ranks=3)
    before = lower_and_check(candidate, context)
    edge = next(iter(used_links(before)))
    run = RunContext.create(tmp_path, 'live-fault', 'local')
    inputs = restricted_inputs(context.inputs, {edge}, run)
    lowering = replace(context, inputs=inputs, topology=load_topology(inputs), lease=StandaloneLease.after(10))
    result = repair_failed_links(before, {edge}, lowering)
    assert result is not None and result.can_execute
    assert edge not in used_links(result)
    assert result.sidecar['request_contract'] == before.sidecar['request_contract']


def test_no_alternative_path_stops_without_returning_the_failed_version(tmp_path):
    from vericcl.live.tuning import repair_failed_links, restricted_inputs, used_links
    from vericcl.topology import load_topology
    (tmp_path / 'inputs').mkdir()
    candidate, context = candidate_fixture(tmp_path / 'inputs', 'broadcast')
    before = lower_and_check(candidate, context)
    edge = next(iter(used_links(before)))
    run = RunContext.create(tmp_path, 'live-fault', 'disconnected')
    inputs = restricted_inputs(context.inputs, {edge}, run)
    result = repair_failed_links(before, {edge}, replace(context, inputs=inputs, topology=load_topology(inputs)))
    assert result is None


def test_background_zero_round_budget_is_respected_and_model_reassessment_is_recorded(tmp_path):
    from vericcl.live.tuning import LiveRepairEngine
    from vericcl.workflow_state import build_budgets
    (tmp_path / 'inputs').mkdir()
    candidate, lowering = candidate_fixture(tmp_path / 'inputs', 'broadcast')
    bundle = lower_and_check(candidate, lowering)
    run = RunContext.create(tmp_path, 'live-background', 'zero-rounds')
    budgets = build_budgets(lowering.inputs)
    budgets.max_tuning_rounds = 0
    engine = LiveRepairEngine(run, budgets)
    event = TriggerEvent('performance', 24, bundle.xml_sha256, {'service:link:0:1': .2})
    value = engine(dict(version=bundle.xml_sha256, bundle=bundle, inputs=lowering.inputs,
        original=lowering.inputs, model=lowering.model, invalid_links=()), (event,), StandaloneLease.after(10))
    assert value.bundle is None and value.status == 'maximum_rounds'
    assert value.model.version != lowering.model.version and budgets.started_rounds == 0
    assert list(run.root.glob('live/background/*/reports/proposal.json'))
