from dataclasses import replace

import pytest

from tests.unit.input.v5_support import request_documents, write_request
from vericcl.input.loader import load_request
from vericcl.topology import load_topology
from vericcl.execution.budget import StandaloneLease
from vericcl.performance.model import CostModel, ServiceCost


def model_for(topology, *, omit_copy=False):
    services, resources = {}, {}
    for key, edge in topology.links.items():
        suffix = '%d:%d' % (key.src_rank, key.dst_rank)
        services['link:' + suffix] = (ServiceCost(1, .01),)
        services['rrc:' + suffix] = (ServiceCost(1.5, .012),)
        resources['link:' + suffix] = 100
    for rank in range(topology.rank_count):
        if not omit_copy:
            services['copy:%d' % rank] = (ServiceCost(.1, .001),)
        services['reduce:%d' % rank] = (ServiceCost(.2, .002),)
    resources.update({key: resource.capacity_bytes_per_us for key, resource in topology.shared_resources.items()})
    return CostModel('lowering-fixture', services, resources)


def candidate_fixture(tmp_path, operator='allreduce', *, method='constructive_trees', ranks=2, inplace=False,
                      layout='separate', channels=1, scratch_bytes=16777216, chain=False, compatibility=True, stages=None):
    from vericcl.solver.constructive import construct_routes, optimize_fixed_routes
    from vericcl.solver.trees.operators import RouteContext
    from vericcl.planner.group_search import HierarchyContext, plan_hierarchy
    docs = request_documents(operator, ranks)
    docs['sketch']['collective']['inplace'] = inplace
    docs['sketch']['hyperparameters']['total_size_bytes'] = ranks * 1024
    docs['sketch']['solver'].update(max_channels=channels, per_model_timeout_s=5)
    docs['sketch']['runtime_compatibility'] = {'enabled': compatibility}
    docs['atom']['synthesis_method'] = method
    if stages is not None:
        docs['atom']['stage_num'] = stages
        docs['topology']['nodes'] = [{'id': 0, 'ranks': list(range(ranks // 2))},
                                     {'id': 1, 'ranks': list(range(ranks // 2, ranks))}]
    docs['runtime'].update(tb_layout=layout, scratch_bytes_per_rank=scratch_bytes)
    if chain:
        docs['topology']['directed_links'] = [e for e in docs['topology']['directed_links'] if e['src'] == e['dst'] + 1]
    inputs = load_request(write_request(tmp_path, docs))
    topology = load_topology(inputs)
    if method == 'constructive_trees':
        context = RouteContext(StandaloneLease.after(30), column_batch_size=2)
        iterator = iter(optimize_fixed_routes(construct_routes(inputs, topology, None, context), context))
    else:
        iterator = iter(plan_hierarchy(inputs, topology, None, HierarchyContext(StandaloneLease.after(30))))
    candidate = next(iterator)
    iterator.close()
    from vericcl.xml.lower import LoweringContext
    return candidate, LoweringContext(inputs, topology, model_for(topology), StandaloneLease.after(20))


@pytest.mark.parametrize('operator', ['broadcast', 'reduce', 'allgather', 'allreduce', 'alltoall', 'reduce_scatter'])
@pytest.mark.parametrize('inplace', [False, True])
def test_six_operators_lower_to_matching_buffers_and_full_prediction(tmp_path, operator, inplace):
    from vericcl.verification.pipeline import lower_and_check
    from vericcl.semantics.version_schedule import replay_version_schedule
    candidate, context = candidate_fixture(tmp_path, operator, inplace=inplace)
    bundle = lower_and_check(candidate, context)
    assert bundle.can_execute, {name: (check.code, check.message) for name, check in bundle.checks.items()}
    assert bundle.prediction.t_pred_us is not None
    replay_version_schedule(bundle.version_schedule, context.inputs)
    assert all(step.count == 1 for block in bundle.program.blocks for step in block.steps)
    assert bundle.sidecar['xml_sha256'] == bundle.xml_sha256
    assert bundle.sidecar['model_version'] == context.model.version
    assert len(bundle.executable.operations) >= len(candidate.schedule.operations)


@pytest.mark.parametrize('operator', ['broadcast', 'allreduce', 'allgather', 'reduce_scatter', 'alltoall', 'reduce'])
def test_hierarchy_uses_the_same_complete_lowering(tmp_path, operator):
    from vericcl.xml.lower import lower_and_check
    candidate, context = candidate_fixture(tmp_path, operator, method='hierarchy')
    bundle = lower_and_check(candidate, context)
    assert bundle.can_execute, {name: check.message for name, check in bundle.checks.items()}
    assert bundle.prediction.model_version == 'lowering-fixture'


def test_network_reduce_is_one_send_and_rrc_pair_without_extra_reduce_tb(tmp_path):
    from vericcl.xml.lower import lower_and_check
    candidate, context = candidate_fixture(tmp_path, 'reduce')
    bundle = lower_and_check(candidate, context)
    assert bundle.can_execute
    network = [op for op in bundle.executable.operations if op.kind == 'rrc']
    steps = [step for block in bundle.program.blocks for step in block.steps]
    assert len([step for step in steps if step.kind == 's']) == len(network)
    assert len([step for step in steps if step.kind == 'rrc']) == len(network)
    assert not any(step.kind == 're' for step in steps)
    assert not any(op.kind == 'reduce' for op in bundle.executable.operations)


@pytest.mark.parametrize('layout', ['separate', 'fused'])
def test_peer_channel_connections_have_one_tb_owner(tmp_path, layout):
    from vericcl.xml.lower import lower_and_check
    candidate, context = candidate_fixture(tmp_path, 'allreduce', ranks=3, channels=2, layout=layout)
    bundle = lower_and_check(candidate, context)
    assert bundle.can_execute, {name: check.message for name, check in bundle.checks.items()}
    owners = []
    for block in bundle.program.blocks:
        if layout == 'separate':
            assert block.send_peer == -1 or block.recv_peer == -1
        for direction, peer in [('send', block.send_peer), ('recv', block.recv_peer)]:
            if peer >= 0:
                owners.append((block.rank, peer, block.channel, direction))
    assert len(owners) == len(set(owners))


def test_scratch_reuse_preserves_previous_read_completion(tmp_path):
    from vericcl.xml.lower import lower_and_check
    candidate, context = candidate_fixture(tmp_path, 'reduce', ranks=3, scratch_bytes=1024, chain=True)
    bundle = lower_and_check(candidate, context)
    assert bundle.can_execute, {name: check.message for name, check in bundle.checks.items()}
    assert max(bundle.program.scratch_slots.values()) <= 1
    assert bundle.sidecar['storage_reuse']
    for reuse in bundle.sidecar['storage_reuse']:
        assert all(bundle.prediction.end_times[reader] <= bundle.prediction.start_times[reuse['writer']]
                   for reader in reuse['previous_readers'])


def test_capacity_failure_retains_candidate_for_repair_and_blocks_execution(tmp_path):
    from vericcl.xml.lower import lower_and_check
    from vericcl.xml.capacity import RuntimeLimits
    candidate, context = candidate_fixture(tmp_path, 'broadcast')
    context = replace(context, limits=RuntimeLimits(steps_per_tb=1))
    bundle = lower_and_check(candidate, context)
    assert not bundle.can_execute and bundle.can_repair
    assert bundle.xml_text and bundle.sidecar
    assert any(issue.code == 'steps_per_tb' for issue in bundle.capacity_report.violations)


def test_disabled_optional_compatibility_does_not_disable_basic_launch_safety(tmp_path):
    from vericcl.xml.lower import lower_and_check
    from vericcl.xml.capacity import RuntimeLimits
    candidate, context = candidate_fixture(tmp_path, 'broadcast', compatibility=False)
    context = replace(context, limits=RuntimeLimits(steps_per_tb=1))
    bundle = lower_and_check(candidate, context)
    assert bundle.checks['compatibility'].status.value == 'not_run'
    assert bundle.checks['launch'].status.value == 'invalid'
    assert not bundle.can_execute


def test_missing_local_cost_does_not_become_zero_in_full_prediction(tmp_path):
    from vericcl.xml.lower import lower_and_check
    candidate, context = candidate_fixture(tmp_path, 'broadcast')
    context = replace(context, model=model_for(context.topology, omit_copy=True))
    bundle = lower_and_check(candidate, context)
    assert any(op.kind == 'copy' for op in bundle.executable.operations)
    assert bundle.prediction.t_pred_us is None
    assert bundle.checks['semantic'].status.value == 'valid'
    assert not bundle.can_execute


@pytest.mark.parametrize('operator', ['broadcast', 'reduce', 'allgather', 'allreduce', 'alltoall', 'reduce_scatter'])
def test_multi_stage_hierarchy_preserves_per_chunk_role_groups(tmp_path, operator):
    from vericcl.xml.lower import lower_and_check
    from vericcl.xml.v5_buffers import normalize_network
    candidate, context = candidate_fixture(tmp_path, operator, method='hierarchy', ranks=4,
                                           stages=4 if operator == 'allreduce' else 3)
    operations = normalize_network(candidate, context)
    assert any(len(op.participants) < 4 for op in operations)
    bundle = lower_and_check(candidate, context)
    assert bundle.can_execute, {name: check.message for name, check in bundle.checks.items()}


def test_explicit_tb_fusion_reuses_same_peer_channel_for_both_directions(tmp_path):
    from vericcl.xml.lower import lower_and_check
    candidate, context = candidate_fixture(tmp_path, 'allgather', layout='fused')
    bundle = lower_and_check(candidate, context)
    assert bundle.can_execute
    assert any(block.send_peer == block.recv_peer >= 0 for block in bundle.program.blocks)


def test_out_of_place_allreduce_keeps_input_bytes_in_separate_storage(tmp_path):
    from vericcl.xml.lower import lower_and_check
    candidate, context = candidate_fixture(tmp_path, 'allreduce')
    bundle = lower_and_check(candidate, context)
    assert bundle.can_execute
    assert all(op.destination.buffer_id != 'input' for op in bundle.version_schedule.operations)


def test_lowering_budget_expiry_remains_a_budget_event(tmp_path, monkeypatch):
    from vericcl.execution.budget import BudgetExhausted
    from vericcl.xml.lower import lower_and_check
    from vericcl.xml import v5_lower
    candidate, context = candidate_fixture(tmp_path, 'broadcast')
    def expire(*args):
        raise BudgetExhausted('offline validation budget')
    monkeypatch.setattr(v5_lower, 'build_storage_plan', expire)
    with pytest.raises(BudgetExhausted):
        lower_and_check(candidate, context)


def test_early_scratch_overwrite_is_rejected_after_real_program_retiming(tmp_path):
    from vericcl.xml.lower import lower_and_check
    from vericcl.xml.v5_lower import evaluate_program
    from vericcl.xml.v5_program import lower_program, emit_native_xml
    candidate, context = candidate_fixture(tmp_path, 'reduce', ranks=3, scratch_bytes=1024, chain=True)
    bundle = lower_and_check(candidate, context)
    assert bundle.can_execute and bundle.storage_plan.reuse
    remove = {item['writer']: set(item['previous_readers']) for item in bundle.storage_plan.reuse}
    broken = replace(bundle.storage_plan, instructions=tuple(replace(op,
        dependencies=op.dependencies - remove.get(op.operation_id, set())) for op in bundle.storage_plan.instructions))
    program = lower_program(broken, context)
    checked = evaluate_program(candidate, broken, program, emit_native_xml(program, context.inputs), context)
    assert not checked.can_execute
    assert checked.checks['semantic'].status.value == 'invalid'
    assert 'reader' in checked.checks['semantic'].message or 'no longer valid' in checked.checks['semantic'].message
