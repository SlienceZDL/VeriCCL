from collections import Counter
from dataclasses import replace

import pytest

from tests.unit.input.v5_support import request_documents, write_request
from vericcl.input.loader import load_request
from vericcl.topology import load_topology
from vericcl.execution.budget import StandaloneLease


def setup(tmp_path, operator, ranks=3, channels=1):
    docs = request_documents(operator, ranks)
    docs['atom']['synthesis_method'] = 'constructive_trees'
    docs['sketch']['hyperparameters']['total_size_bytes'] = ranks * 1024
    docs['sketch']['solver'].update(max_channels=channels, per_model_timeout_s=5)
    inputs = load_request(write_request(tmp_path, docs))
    return inputs, load_topology(inputs)


@pytest.mark.parametrize('operator', ['broadcast', 'reduce', 'allgather', 'allreduce', 'alltoall', 'reduce_scatter'])
def test_six_operators_route_every_integer_slice_and_preserve_outputs(tmp_path, operator):
    from vericcl.solver.constructive import construct_routes, optimize_fixed_routes
    from vericcl.solver.trees.operators import RouteContext, check_routed_semantics
    inputs, topology = setup(tmp_path, operator)
    context = RouteContext(StandaloneLease.after(30), column_batch_size=4)
    bundle = construct_routes(inputs, topology, None, context)
    check_routed_semantics(bundle)
    assert all(len(batch.demand_ids) == batch.slice_count for batch in bundle.batches)
    iterator = iter(optimize_fixed_routes(bundle, context))
    candidate = next(iterator)
    iterator.close()
    assert candidate.model_calls >= 1
    assert candidate.solver_result.status in {'optimal', 'feasible', 'time_limit'}
    assert candidate.incumbent_preserved and candidate.communication_time_us <= candidate.initial_time_us + 1e-7
    assert not candidate.runtime_checked and not candidate.full_search_proven
    check_routed_semantics(candidate.bundle)


def test_reduction_dual_restores_actual_directed_edges(tmp_path):
    from vericcl.solver.constructive import construct_routes
    from vericcl.solver.trees.operators import RouteContext
    inputs, topology = setup(tmp_path, 'reduce')
    from vericcl.topology.model import LinkKey
    topology = replace(topology, links={key: value for key, value in topology.links.items()
                                        if key in {LinkKey(2, 1), LinkKey(1, 0)}}, isomorphism_signature='')
    bundle = construct_routes(inputs, topology, None, RouteContext(StandaloneLease.after(20)))
    assert {op.edge for op in bundle.operations} == {(2, 1), (1, 0)}
    for op in bundle.operations:
        assert op.kind == 'rrc'
        if op.edge == (1, 0):
            assert op.contributors == frozenset({1 * 3 + op.logical_position, 2 * 3 + op.logical_position})
            assert op.dependencies


def test_allreduce_charges_reduce_and_distribution_to_same_shared_capacity(tmp_path):
    from vericcl.solver.constructive import construct_routes
    from vericcl.solver.trees.operators import RouteContext
    docs = request_documents('allreduce', 2)
    docs['atom']['synthesis_method'] = 'constructive_trees'
    docs['sketch']['hyperparameters']['total_size_bytes'] = 2048
    for edge in docs['topology']['directed_links']:
        edge['resources'] = ['fabric']
    docs['topology']['shared_resources'] = [{'id': 'fabric', 'capacity_bytes_per_us': 100, 'links': [[0, 1], [1, 0]]}]
    inputs = load_request(write_request(tmp_path, docs))
    bundle = construct_routes(inputs, load_topology(inputs), None, RouteContext(StandaloneLease.after(20)))
    assert bundle.resource_load_bytes['fabric'] == 4 * 1024
    assert bundle.bandwidth_time_us >= 40.96 - 1e-8
    assert {op.kind for op in bundle.operations} == {'send', 'rrc'}
    assert bundle.routing_scope == 'integer_route_columns'


def test_alltoall_has_exact_source_destination_coverage(tmp_path):
    from vericcl.solver.constructive import construct_routes
    from vericcl.solver.trees.operators import RouteContext
    inputs, topology = setup(tmp_path, 'alltoall')
    bundle = construct_routes(inputs, topology, None, RouteContext(StandaloneLease.after(20)))
    assert len(bundle.demands) == 9
    assert {(item.source_rank, item.destinations[0]) for item in bundle.demands} == {(a, b) for a in range(3) for b in range(3)}
    assert len({id for batch in bundle.batches for id in batch.demand_ids}) == 9


def test_route_batches_are_independent_of_execution_channels(tmp_path):
    from vericcl.solver.constructive import construct_routes
    from vericcl.solver.trees.operators import RouteContext
    inputs, topology = setup(tmp_path, 'broadcast')
    one = construct_routes(inputs, topology, None, RouteContext(StandaloneLease.after(20)))
    many = construct_routes(replace(inputs, solver=replace(inputs.solver, max_channels=3)), topology,
                            None, RouteContext(StandaloneLease.after(20)))
    assert [(b.demand_ids, b.route) for b in one.batches] == [(b.demand_ids, b.route) for b in many.batches]
    assert one.bandwidth_time_us == many.bandwidth_time_us


def test_finite_column_batch_expands_in_candidate_stream(tmp_path):
    from vericcl.solver.constructive import construct_routes, optimize_fixed_routes
    from vericcl.solver.trees.operators import RouteContext
    inputs, topology = setup(tmp_path, 'allreduce', 2)
    context = RouteContext(StandaloneLease.after(20), column_batch_size=1)
    bundle = construct_routes(inputs, topology, None, context)
    assert not bundle.column_space_complete
    candidates = list(optimize_fixed_routes(bundle, context))
    assert candidates[-1].bundle.column_space_complete
    assert candidates[-1].bundle.generated_columns > bundle.generated_columns
    assert not candidates[-1].full_search_proven


def test_forbidden_direct_link_allows_detour(tmp_path):
    from vericcl.solver.constructive import construct_routes
    from vericcl.solver.trees.operators import RouteContext
    from vericcl.input.models import AtomConstraints, ForbiddenTransfer
    inputs, topology = setup(tmp_path, 'broadcast')
    inputs = replace(inputs, atom_constraints=AtomConstraints(None, (ForbiddenTransfer(0, 0, 1, 0),)))
    bundle = construct_routes(inputs, topology, None, RouteContext(StandaloneLease.after(20), column_batch_size=4))
    assert not any(op.edge == (0, 1) and 0 in op.contributors for op in bundle.operations)
    assert any(op.edge == (2, 1) and 0 in op.contributors for op in bundle.operations)


def test_bundle_checker_rejects_missing_data_dependency(tmp_path):
    from vericcl.solver.constructive import construct_routes
    from vericcl.solver.trees.operators import RouteContext, check_routed_semantics
    from vericcl.topology.model import LinkKey
    from vericcl.errors import SemanticError
    inputs, topology = setup(tmp_path, 'broadcast')
    topology = replace(topology, links={key: edge for key, edge in topology.links.items()
                                        if key in {LinkKey(0, 1), LinkKey(1, 2)}}, isomorphism_signature='')
    bundle = construct_routes(inputs, topology, None, RouteContext(StandaloneLease.after(10)))
    broken = replace(bundle, operations=tuple(replace(op, dependencies=frozenset()) for op in bundle.operations))
    with pytest.raises(SemanticError, match='dependenc'):
        check_routed_semantics(broken)


def test_fixed_route_timing_export_has_no_hidden_idle_releases(tmp_path):
    from vericcl.solver.constructive import construct_routes, optimize_fixed_routes
    from vericcl.solver.trees.operators import RouteContext
    inputs, topology = setup(tmp_path, 'allreduce', channels=2)
    context = RouteContext(StandaloneLease.after(20), column_batch_size=2)
    stream = iter(optimize_fixed_routes(construct_routes(inputs, topology, None, context), context))
    candidate = next(stream)
    stream.close()
    for identifier, start in candidate.schedule.start_us.items():
        assert start == max((candidate.schedule.finish_us[p] for p in candidate.schedule.dependencies[identifier]), default=0)
    operations = candidate.bundle.operations
    for left in operations:
        for right in operations:
            if left.operation_id >= right.operation_id:
                continue
            same_accumulator = left.kind == right.kind == 'rrc' and left.demand_id == right.demand_id and left.edge[1] == right.edge[1]
            same_lane = left.edge == right.edge and candidate.schedule.channels[left.operation_id] == candidate.schedule.channels[right.operation_id]
            if same_lane or same_accumulator:
                assert (candidate.schedule.finish_us[left.operation_id] <= candidate.schedule.start_us[right.operation_id] + 1e-8
                        or candidate.schedule.finish_us[right.operation_id] <= candidate.schedule.start_us[left.operation_id] + 1e-8)


def test_single_rank_identity_still_calls_required_timing_solver(tmp_path):
    from vericcl.solver.constructive import construct_routes, optimize_fixed_routes
    from vericcl.solver.trees.operators import RouteContext
    inputs, topology = setup(tmp_path, 'broadcast', ranks=1)
    context = RouteContext(StandaloneLease.after(10))
    candidates = list(optimize_fixed_routes(construct_routes(inputs, topology, None, context), context))
    assert len(candidates) == 1
    assert candidates[0].communication_time_us == 0
    assert context.model_calls == 1


def test_allreduce_joint_integer_routes_match_small_exhaustive_oracle(tmp_path):
    from fractions import Fraction
    from itertools import product
    from vericcl.solver.constructive import construct_routes
    from vericcl.solver.trees.operators import RouteContext
    docs = request_documents('allreduce', 3)
    docs['atom']['synthesis_method'] = 'constructive_trees'
    docs['sketch']['hyperparameters'].update(total_size_bytes=2048, slice_size_bytes=1024)
    weights = {(0, 1): 1, (0, 2): 4, (1, 0): 3, (1, 2): 1, (2, 0): 1, (2, 1): 2}
    for edge in docs['topology']['directed_links']:
        edge['beta_us_per_byte'] = weights[edge['src'], edge['dst']] / 1024
    inputs = load_request(write_request(tmp_path, docs))
    bundle = construct_routes(inputs, load_topology(inputs), None, RouteContext(StandaloneLease.after(20), column_batch_size=100))
    # Enumerate the three out-trees for each root independently of production
    # column generation; reductions are their reversed directed edges.
    columns = []
    for root in range(3):
        a, b = [r for r in range(3) if r != root]
        trees = [((root, a), (root, b)), ((root, a), (a, b)), ((root, b), (b, a))]
        columns.extend(tuple((y, x) for x, y in red) + send for red, send in product(trees, repeat=2))
    best = min(max(Counter(first + second)[edge] * weight for edge, weight in weights.items())
               for first, second in product(columns, repeat=2))
    assert Fraction(*bundle.bandwidth_time_ratio) == best
    assert bundle.column_space_complete
    assert bundle.routing_solver_result.numerical_optimal
    assert not bundle.routing_exact_optimal


def test_one_timing_model_budget_does_not_claim_parent_budget_exhaustion(tmp_path, monkeypatch):
    from vericcl.solver.constructive import construct_routes, optimize_fixed_routes
    from vericcl.solver.trees.operators import RouteContext
    from vericcl.solver.trees import timing
    from vericcl.execution.budget import BudgetExhausted
    inputs, topology = setup(tmp_path, 'allreduce', 2)
    context = RouteContext(StandaloneLease.after(20), column_batch_size=1)
    bundle = construct_routes(inputs, topology, None, context)
    original = timing.solve_fixed_routes
    calls = []
    def one_expired_model(bundle, context):
        calls.append(1)
        if len(calls) == 1:
            raise BudgetExhausted('single model construction budget')
        return original(bundle, context)
    monkeypatch.setattr(timing, 'solve_fixed_routes', one_expired_model)
    stream = optimize_fixed_routes(bundle, context)
    candidates = list(stream)
    assert candidates
    assert stream.stop_reason != 'initial_synthesis_budget'
    assert any(record['status'] == 'timing_model_budget' for record in context.records)


def test_checked_seed_survives_later_order_model_budget(tmp_path, monkeypatch):
    from vericcl.solver.constructive import construct_routes, optimize_fixed_routes
    from vericcl.solver.trees.operators import RouteContext
    from vericcl.solver.trees import timing
    from vericcl.execution.budget import BudgetExhausted
    inputs, topology = setup(tmp_path, 'broadcast', ranks=2)
    context = RouteContext(StandaloneLease.after(20), column_batch_size=100)
    bundle = construct_routes(inputs, topology, None, context)
    calls = []
    def expired_order_model(bundle, context):
        calls.append('full_order_search')
        raise BudgetExhausted('order model construction deadline')
    monkeypatch.setattr(timing, 'solve_fixed_routes', expired_order_model)
    stream = optimize_fixed_routes(bundle, context)
    candidates = list(stream)
    assert len(candidates) == 1
    assert calls == ['full_order_search']
    assert candidates[0].solver_result.primal is not None
    assert not candidates[0].full_search_proven
    assert any(record.get('scope') == 'fixed_route_seed_timing' for record in context.records)
    assert any(record.get('status') == 'timing_model_budget' for record in context.records)
    assert stream.stop_reason != 'initial_synthesis_budget'


def test_seed_requires_native_solver(tmp_path, monkeypatch):
    from vericcl.errors import SolverUnavailableError
    from vericcl.solver.constructive import construct_routes, optimize_fixed_routes
    from vericcl.solver.scip_api import ScipAdapter
    from vericcl.solver.trees.operators import RouteContext
    inputs, topology = setup(tmp_path, 'broadcast', ranks=2)
    context = RouteContext(StandaloneLease.after(20))
    bundle = construct_routes(inputs, topology, None, context)
    def unavailable(*args, **kwargs):
        raise SolverUnavailableError('required solver unavailable')
    monkeypatch.setattr(ScipAdapter, 'create_model', unavailable)
    with pytest.raises(SolverUnavailableError, match='required solver unavailable'):
        next(iter(optimize_fixed_routes(bundle, context)))


def test_parent_budget_during_solver_order_replay_propagates(tmp_path, monkeypatch):
    from vericcl.solver.constructive import construct_routes
    from vericcl.solver.trees.operators import RouteContext
    from vericcl.solver.trees import timing
    from vericcl.execution.budget import BudgetExhausted
    inputs, topology = setup(tmp_path, 'broadcast', ranks=2)
    context = RouteContext(StandaloneLease.after(20))
    bundle = construct_routes(inputs, topology, None, context)
    def expired_replay(*args, **kwargs):
        raise BudgetExhausted('parent expired during replay')
    monkeypatch.setattr(timing, '_earliest', expired_replay)
    with pytest.raises(BudgetExhausted, match='parent expired during replay'):
        timing.solve_fixed_routes(bundle, context)


def test_single_channel_full_model_can_reverse_seed_resource_order(tmp_path):
    from vericcl.solver.constructive import construct_routes
    from vericcl.solver.trees.operators import RouteContext, Route, RouteBatch, _physical_operation
    from vericcl.solver.trees import timing
    docs = request_documents('broadcast', 3)
    docs['atom']['synthesis_method'] = 'constructive_trees'
    docs['sketch']['hyperparameters'].update(total_size_bytes=2048, slice_size_bytes=1024)
    docs['sketch']['solver'].update(max_channels=1, per_model_timeout_s=5)
    for edge in docs['topology']['directed_links']:
        edge['alpha_us'] = 0.
        edge['beta_us_per_byte'] = (10. if (edge['src'], edge['dst']) == (1, 2) else 1.) / 1024
    inputs = load_request(write_request(tmp_path, docs))
    topology = load_topology(inputs)
    context = RouteContext(StandaloneLease.after(20))
    bundle = construct_routes(inputs, topology, None, context)
    routes = (Route(0, send_edges=((0, 1), (0, 2))), Route(0, send_edges=((0, 1), (1, 2))))
    operations = tuple(op for demand, route in zip(bundle.demands, routes)
                       for op in _physical_operation(inputs, topology, demand, route))
    batches = tuple(RouteBatch('batch-%d' % index, (demand.demand_id,), route, ())
                    for index, (demand, route) in enumerate(zip(bundle.demands, routes)))
    bundle = replace(bundle, operations=operations, batches=batches)
    seed = timing.solve_fixed_seed(bundle, context)
    full = timing.solve_fixed_routes(bundle, context)
    assert seed.communication_time_us == pytest.approx(12.)
    assert full.communication_time_us == pytest.approx(11.)
    assert full.schedule.start_us[operations[2].operation_id] < full.schedule.start_us[operations[0].operation_id]
    assert not full.full_search_proven
