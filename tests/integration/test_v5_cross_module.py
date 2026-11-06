"""Checks spanning original-byte constraints, lowering, models and evidence."""
from dataclasses import replace
import json

import pytest

from tools.generate_examples import example_documents, write_documents


@pytest.mark.parametrize('slice_bytes', [512, 1024])
def test_original_byte_edge_ban_survives_partition_and_final_xml(tmp_path, slice_bytes):
    from vericcl.input.loader import load_request
    from vericcl.topology import load_topology
    from vericcl.planner.group_search import HierarchyContext, plan_hierarchy
    from vericcl.execution.budget import StandaloneLease
    from tests.integration.test_lowering_v5 import model_for
    from vericcl.xml.lower import LoweringContext, lower_and_check
    from tests.e2e.v5_reference import replay_xml_contributions
    documents = example_documents('broadcast', ranks=3)
    documents['sketch']['hyperparameters'].update(total_size_bytes=2048, slice_size_bytes=slice_bytes,
                                                 reference_slice_size_bytes=512)
    documents['atom'].update(stage_num=1, forbidden_transfers=[[1, 0, 1, 0]])
    inputs = load_request(write_documents(tmp_path / 'input', documents)); topology = load_topology(inputs)
    stream = iter(plan_hierarchy(inputs, topology, None, HierarchyContext(StandaloneLease.after(25))))
    candidate = next(stream); stream.close()
    bundle = lower_and_check(candidate, LoweringContext(inputs, topology, model_for(topology), StandaloneLease.after(10)))
    assert bundle.can_execute, bundle.checks
    # The ban removes only the direct edge for bytes [512,1024). Other bytes
    # remain free to use it, and the affected data can arrive via rank 2.
    from vericcl.xml.v5_buffers import normalize_network
    affected_edges = set()
    for operation in normalize_network(candidate, LoweringContext(inputs, topology, model_for(topology), StandaloneLease.after(5))):
        # Contributor IDs use the effective partition. Bind the original ban
        # independently in byte coordinates, including partially overlapping
        # larger slices, instead of reusing the production constraint binder.
        affected = any(origin == 0 and position * slice_bytes < 1024 and
                       (position + 1) * slice_bytes > 512
                       for origin, position in
                       (divmod(member, 2048 // slice_bytes) for member in operation.contributors))
        if affected and operation.stage_id == 0:
            assert (operation.src_rank, operation.dst_rank) != (0, 1)
            affected_edges.add((operation.src_rank, operation.dst_rank))
    assert {(0, 2), (2, 1)} <= affected_edges
    path = tmp_path / 'schedule.xml'; path.write_text(bundle.xml_text)
    replay_xml_contributions(path, 'broadcast', ranks=3, slices=2048 // slice_bytes)


def test_current_model_rechecks_same_xml_and_blocks_cross_model_gain(tmp_path):
    from tests.integration.test_lowering_v5 import candidate_fixture
    from vericcl.xml.lower import lower_and_check
    from vericcl.xml.v5_lower import evaluate_program
    from vericcl.workflow_v5 import accept_candidate
    candidate, context = candidate_fixture(tmp_path)
    old = lower_and_check(candidate, context)
    updated = replace(context.model, version='changed', services={key: tuple(replace(piece,
        alpha_us=piece.alpha_us * 2, beta_us_per_byte=piece.beta_us_per_byte * 2) for piece in values)
        for key, values in context.model.services.items()})
    current = evaluate_program(candidate, old.storage_plan, old.program, old.xml_text, replace(context, model=updated))
    assert current.can_execute and current.xml_sha256 == old.xml_sha256
    assert current.prediction.t_pred_us > old.prediction.t_pred_us
    assert not accept_candidate(current, old, {}).accepted


def test_log_failure_after_timeout_keeps_failure_and_incumbent_evidence(tmp_path, monkeypatch):
    from vericcl.execution.context import RunContext, StorageFailure, finalize_run
    from vericcl.execution.status import RunResult
    context = RunContext.create(tmp_path, 'failure', 'r')
    context.write_json('best/accepted.json', {'xml_sha256': 'already-accepted', 'model_version': 'm1'})
    context.write_json('jobs/timed-out/result.json', {'status': 'timeout', 'cleanup_complete': False})
    original = context.write_json
    def fail_summary(relative, value, **options):
        if relative == 'run-summary.json':
            raise StorageFailure('injected disk full during timeout finalization')
        return original(relative, value, **options)
    monkeypatch.setattr(context, 'write_json', fail_summary)
    result = finalize_run(context, RunResult(False, 'incomplete', stop_reason='time_limit', failure_category='offline'))
    assert not result.complete and result.exit_code == 7
    assert json.loads((context.artifact('best/accepted.json')).read_text())['xml_sha256'] == 'already-accepted'
    assert json.loads((context.artifact('jobs/timed-out/result.json')).read_text())['cleanup_complete'] is False
