from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.integration.test_lowering_v5 import candidate_fixture
from vericcl.execution.budget import StandaloneLease
from vericcl.execution.context import RunContext
from vericcl.input.schema import _freeze, LIVE_TIMEOUTS
from vericcl.live.triggers import TriggerEvent
from vericcl.workflow_state import build_budgets, plain
from vericcl.xml.lower import lower_and_check


def fixture(tmp_path, ranks):
    candidate, lowering = candidate_fixture(tmp_path, 'broadcast', ranks=ranks)
    config = plain(lowering.inputs.request_config)
    config['sketch']['live'].update(enabled=True, **{key: 5 for key in LIVE_TIMEOUTS})
    config['runtime']['control'] = {'bind_address': '127.0.0.1', 'port': 12345, 'unix_socket': '/home/example-user/control-test.sock'}
    inputs = replace(lowering.inputs, request_config=_freeze(config))
    lowering = replace(lowering, inputs=inputs)
    bundle = lower_and_check(candidate, lowering)
    context = RunContext.create(tmp_path, 'adaptive', 'run')
    workflow = SimpleNamespace(original=inputs, best_inputs=inputs, best=bundle, model=lowering.model,
                               budgets=build_budgets(inputs))
    prepared = []
    session = SimpleNamespace(context=context, inputs=inputs, bundle=bundle, settings=config['sketch']['live'],
        lease=StandaloneLease.after(20), prepare_version=lambda bundle, inputs, **options: prepared.append(bundle.xml_sha256) or
        {'prepared': True, 'activated': False, 'version_id': bundle.xml_sha256})
    return workflow, session, prepared


def test_fault_blocks_next_request_until_legal_replacement_is_prepared(tmp_path):
    from vericcl.live.adaptive import AdaptiveLiveController
    from vericcl.live.tuning import used_links
    workflow, session, prepared = fixture(tmp_path, 3)
    control = AdaptiveLiveController(session, workflow)
    edge = next(iter(used_links(control.bundle)))
    control.submit(TriggerEvent('fault', 0, control.bundle.xml_sha256, invalid_links=(edge,),
                                evidence={'source': 'fault_injection'}))
    try:
        control.before_request()
        assert prepared == [control.bundle.xml_sha256]
        assert edge not in used_links(control.bundle)
        assert control.invalid_links == {edge}
    finally:
        assert control.close(5)


def test_no_legal_replacement_stops_group_and_preserves_failed_link_exclusion(tmp_path):
    from vericcl.live.adaptive import AdaptiveLiveController
    from vericcl.live.tuning import used_links
    workflow, session, prepared = fixture(tmp_path, 2)
    control = AdaptiveLiveController(session, workflow)
    edge = next(iter(used_links(control.bundle)))
    control.submit(TriggerEvent('fault', 0, control.bundle.xml_sha256, invalid_links=(edge,),
                                evidence={'source': 'fault_injection'}))
    try:
        with pytest.raises(RuntimeError, match='legal replacement'):
            control.before_request()
        assert not prepared and edge in control.invalid_links
    finally:
        assert control.close(5)


def test_foreground_revalidation_uses_the_shared_offline_budget(tmp_path):
    from vericcl.live.adaptive import AdaptiveLiveController
    from vericcl.live.tuning import LiveProposal
    workflow, session, prepared = fixture(tmp_path, 2)
    control = AdaptiveLiveController(session, workflow)
    proposal = LiveProposal(workflow.best, workflow.best_inputs, workflow.model, 'performance_proposal', (), 'unused')
    assert not control._commit_proposal(proposal)  # Identical XML has no gain.
    assert workflow.budgets.consumed['offline'] > 0
    assert not prepared and control.close(1)


def test_exhausted_revalidation_budget_keeps_the_current_valid_version(tmp_path):
    from vericcl.live.adaptive import AdaptiveLiveController
    from vericcl.live.tuning import LiveProposal
    workflow, session, prepared = fixture(tmp_path, 2)
    control = AdaptiveLiveController(session, workflow)
    workflow.budgets.consumed['offline'] = workflow.budgets.stage_limits['offline'][1]
    proposal = LiveProposal(workflow.best, workflow.best_inputs, workflow.model, 'performance_proposal', (), 'unused')
    assert not control._commit_proposal(proposal)
    assert control.last_failure == 'revalidation_budget_exhausted'
    control.before_request()
    assert not prepared and control.close(1)
