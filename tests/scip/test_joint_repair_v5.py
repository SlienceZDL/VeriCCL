from dataclasses import replace

import pytest

from tests.unit.input.v5_support import request_documents, write_request
from vericcl.execution.budget import StandaloneLease
from vericcl.input.loader import load_request
from vericcl.topology import load_topology
from vericcl.performance.model import CostModel, ServiceCost
from vericcl.xml.lower import LoweringContext, lower_and_check


def repair_fixture(tmp_path, *, channels=1):
    from vericcl.xml.v5_model import NetworkCandidate, NetworkOperation
    from vericcl.tuning.strategies import RepairContext
    docs = request_documents('broadcast', 3)
    docs['sketch']['solver'].update(max_channels=channels, per_model_timeout_s=5)
    docs['atom']['synthesis_method'] = 'constructive_trees'
    inputs = load_request(write_request(tmp_path, docs))
    topology = load_topology(inputs)
    services, resources = {}, {}
    for key in topology.links:
        beta = .1 if (key.src_rank, key.dst_rank) == (0, 1) else .01
        suffix = '%d:%d' % (key.src_rank, key.dst_rank)
        services['link:' + suffix] = (ServiceCost(1, beta),)
        services['rrc:' + suffix] = (ServiceCost(1.5, beta),)
        resources['link:' + suffix] = 1 / beta
    for rank in range(3):
        services['copy:%d' % rank] = (ServiceCost(.1, .001),)
        services['reduce:%d' % rank] = (ServiceCost(.2, .002),)
    model = CostModel('repair-fixture', services, resources)
    operations = tuple(NetworkOperation('slice%d-to%d' % (s, dst), 'send', 0, dst, 0, 0,
        frozenset({s}), frozenset({s}), (0, 1, 2), frozenset(), frozenset(), float(s))
        for s in range(4) for dst in (1, 2))
    candidate = NetworkCandidate(inputs.rank_count, inputs.hyperparameters.slice_count,
                                 inputs.hyperparameters.slice_size_bytes, operations, 'repair-test')
    lowering = LoweringContext(inputs, topology, model, StandaloneLease.after(30))
    bundle = lower_and_check(candidate, lowering)
    assert bundle.can_execute, {key: check.message for key, check in bundle.checks.items()}
    return bundle, RepairContext(lowering, candidate_batch_size=4)


def test_two_slices_rebalance_jointly_and_preserve_every_outside_decision(tmp_path):
    from vericcl.verification.bottlenecks import StrategyRecord
    from vericcl.tuning.strategies import build_domain, solve_repair, outside_invariants
    bundle, context = repair_fixture(tmp_path)
    ids = ('slice0-to1', 'slice1-to1')
    strategy = StrategyRecord('allocation', ids, paths=tuple((key, (0, 2, 1)) for key in ids))
    domain = build_domain(None, strategy)
    result = solve_repair(bundle, domain, strategy, context)
    assert result.status == 'improved', result
    assert result.candidate.can_execute
    assert result.candidate.prediction.t_pred_us < bundle.prediction.t_pred_us
    assert result.model_calls >= 1
    assert outside_invariants(bundle, result.candidate, domain).valid
    assert result.acceptance_basis == 'same_model_complete_prediction'
    assert result.domain.operation_ids == frozenset(ids)


def test_domain_escape_is_rejected_before_native_solver(tmp_path):
    from vericcl.verification.bottlenecks import StrategyRecord
    from vericcl.tuning.strategies import build_domain, solve_repair
    bundle, context = repair_fixture(tmp_path)
    allowed = StrategyRecord('path', ('slice0-to1',), paths=(('slice0-to1', (0, 2, 1)),))
    escape = replace(allowed, paths=allowed.paths + (('slice1-to1', (0, 2, 1)),))
    result = solve_repair(bundle, build_domain(None, allowed), escape, context)
    assert result.status == 'invalid_strategy' and result.model_calls == 0
    assert result.candidate is None


def test_outside_work_can_move_earlier_after_hot_link_load_is_removed(tmp_path):
    from vericcl.verification.bottlenecks import StrategyRecord
    from vericcl.tuning.strategies import build_domain, solve_repair
    bundle, context = repair_fixture(tmp_path)
    strategy = StrategyRecord('path', ('slice0-to1',), paths=(('slice0-to1', (0, 2, 1)),))
    result = solve_repair(bundle, build_domain(None, strategy), strategy, context)
    assert result.status == 'improved'
    assert result.candidate.prediction.end_times['slice3-to1'] < bundle.prediction.end_times['slice3-to1']


def test_greedy_valid_improvement_still_enters_required_native_model(tmp_path, monkeypatch):
    from vericcl.verification.bottlenecks import StrategyRecord
    from vericcl.tuning.strategies import build_domain, solve_repair
    from vericcl.solver.scip_api import ScipAdapter
    bundle, context = repair_fixture(tmp_path)
    strategy = StrategyRecord('path', ('slice0-to1',), paths=(('slice0-to1', (0, 2, 1)),))
    original, calls = ScipAdapter.solve, []
    def counted(model, lease):
        calls.append(model.getProbName())
        return original(model, lease)
    monkeypatch.setattr(ScipAdapter, 'solve', counted)
    result = solve_repair(bundle, build_domain(None, strategy), strategy, context)
    assert result.status == 'improved'
    assert calls and all('event' in name for name in calls)
    assert result.solver_evidence and result.solver_evidence[-1]['eager_start_enforced']


def test_joint_mode_can_choose_paths_for_multiple_slices(tmp_path):
    from vericcl.verification.bottlenecks import StrategyRecord
    from vericcl.tuning.strategies import build_domain, solve_repair
    bundle, context = repair_fixture(tmp_path)
    context.mode = 'joint'
    strategy = StrategyRecord('joint', ('slice0-to1', 'slice1-to1'))
    result = solve_repair(bundle, build_domain(None, strategy), strategy, context)
    assert result.status == 'improved'
    assert result.model_calls > 0
    assert result.candidate.prediction.model_version == bundle.prediction.model_version
    assert result.search_complete is False


def test_local_objective_gain_without_complete_time_gain_is_rejected(tmp_path):
    from vericcl.verification.bottlenecks import StrategyRecord
    from vericcl.tuning.strategies import build_domain, solve_repair
    bundle, context = repair_fixture(tmp_path, channels=2)
    # Only a fast-link chunk changes channel; the fixed slow-link tail still wins.
    strategy = StrategyRecord('channel', ('slice0-to2',), channels=(('slice0-to2', 1),))
    result = solve_repair(bundle, build_domain(None, strategy), strategy, context)
    assert result.status == 'no_improvement'
    assert result.candidate is None and result.model_calls > 0
    assert result.incumbent.xml_sha256 == bundle.xml_sha256


def test_budget_exhaustion_retains_incumbent(tmp_path):
    from vericcl.verification.bottlenecks import StrategyRecord
    from vericcl.tuning.strategies import build_domain, solve_repair
    bundle, context = repair_fixture(tmp_path)
    context.lowering.lease.cancel()
    strategy = StrategyRecord('joint', ('slice0-to1',))
    result = solve_repair(bundle, build_domain(None, strategy), strategy, context)
    assert result.status == 'budget_exhausted' and result.incumbent is bundle


def test_tb_order_change_improves_a_downstream_tail(tmp_path):
    from vericcl.xml.v5_model import NetworkOperation
    from vericcl.verification.bottlenecks import StrategyRecord
    from vericcl.tuning.strategies import build_domain, solve_repair
    bundle, context = repair_fixture(tmp_path)
    model, _ = context.lowering.model.with_updates('order-tail',
        services={'link:0:1': (ServiceCost(1, .01),), 'link:2:1': (ServiceCost(1, .2),)},
        resources={'link:0:1': 100, 'link:2:1': 5})
    context.lowering = replace(context.lowering, model=model)
    original = bundle.source
    prefix = NetworkOperation('relay-prefix', 'send', 0, 2, 0, 0, frozenset({0}), frozenset({0}),
                              (0, 1, 2), frozenset(), frozenset(), 100.)
    operations = tuple(replace(op, src_rank=2, dependencies=frozenset({'relay-prefix'}), priority_time=101.)
                       if op.operation_id == 'slice0-to1' else op for op in original.operations) + (prefix,)
    bundle = lower_and_check(replace(original, operations=operations), context.lowering)
    assert bundle.can_execute
    strategy = StrategyRecord('order', ('relay-prefix', 'slice0-to2'), orders=(('relay-prefix', 'slice0-to2'),))
    result = solve_repair(bundle, build_domain(None, strategy), strategy, context)
    assert result.status == 'improved', result
    assert result.candidate.prediction.end_times['slice0-to1'] < bundle.prediction.end_times['slice0-to1']


def test_native_event_model_matches_measured_asymmetric_shared_rates(tmp_path):
    from vericcl.performance.profiles import RateProfile
    from vericcl.tuning.event_columns import solve_event_columns
    bundle, context = repair_fixture(tmp_path, channels=2)
    source = bundle.source
    changed = replace(source, operations=tuple(replace(op, channel=op.contributors.__iter__().__next__() % 2)
                                             if op.dst_rank == 1 else op for op in source.operations))
    profile = RateProfile('asymmetric', ('send:0:1:c0', 'send:0:1:c1'),
                          {'send:0:1:c0': 4., 'send:0:1:c1': 6.}, resources=('link:0:1',),
                          measured_at='2026-11-03T00:00:00Z')
    model, _ = context.lowering.model.with_updates('measured-sharing', profiles=(profile,))
    context.lowering = replace(context.lowering, model=model)
    original = lower_and_check(source, context.lowering)
    alternate = lower_and_check(changed, context.lowering)
    assert original.can_execute and alternate.can_execute
    selected, native = solve_event_columns((original, alternate), context)
    assert selected is not None
    assert native.primal == pytest.approx(min(original.prediction.t_pred_us, alternate.prediction.t_pred_us))
    assert 'profile:asymmetric' in alternate.prediction.used_components


def test_inplace_overwrite_guards_are_mandatory_during_order_search(tmp_path):
    from tests.integration.test_lowering_v5 import candidate_fixture
    from vericcl.verification.bottlenecks import mandatory_dependencies
    from vericcl.xml.v5_buffers import canonical_region
    candidate, lowering = candidate_fixture(tmp_path, 'allgather', ranks=3, inplace=True)
    bundle = lower_and_check(candidate, lowering)
    assert bundle.can_execute
    storage = bundle.storage_plan
    mandatory = mandatory_dependencies(storage, lowering.inputs)
    checked = 0
    for writer in storage.instructions:
        target = canonical_region(storage.addresses[writer.output], lowering.inputs)
        for key in storage.initial_ids:
            if target.overlaps(canonical_region(storage.addresses[key], lowering.inputs)):
                for reader in storage.instructions:
                    if key in reader.inputs and writer.operation_id != reader.operation_id:
                        assert reader.operation_id in mandatory[writer.operation_id]
                        checked += 1
    # AllGather's per-rank original input is retained in its own output slot;
    # use AllReduce below when that identity creates no final overwrite.
    if not checked:
        other = tmp_path / 'allreduce'; other.mkdir()
        candidate, lowering = candidate_fixture(other, 'allreduce', ranks=3, inplace=True)
        bundle = lower_and_check(candidate, lowering)
        storage = bundle.storage_plan
        mandatory = mandatory_dependencies(storage, lowering.inputs)
        for writer in storage.instructions:
            target = canonical_region(storage.addresses[writer.output], lowering.inputs)
            for key in storage.initial_ids:
                if target.overlaps(canonical_region(storage.addresses[key], lowering.inputs)):
                    for reader in storage.instructions:
                        if key in reader.inputs and writer.operation_id != reader.operation_id:
                            assert reader.operation_id in mandatory[writer.operation_id]
                            checked += 1
    assert checked > 0
