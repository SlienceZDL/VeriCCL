from dataclasses import replace
import itertools

import pytest

from tests.unit.input.v5_support import request_documents, write_request
from vericcl.input.loader import load_request
from vericcl.topology import load_topology


def setup(tmp_path, operator='allreduce', ranks=4, *, stages=None):
    docs = request_documents(operator, ranks)
    docs['topology']['nodes'] = [{'id': 0, 'ranks': list(range(ranks // 2))},
                                 {'id': 1, 'ranks': list(range(ranks // 2, ranks))}]
    docs['sketch']['solver'].update(max_channels=1, per_model_timeout_s=5)
    if stages is not None:
        docs['atom']['stage_num'] = stages
    inputs = load_request(write_request(tmp_path, docs))
    return inputs, load_topology(inputs)


@pytest.mark.parametrize('operator', ['broadcast', 'reduce', 'allgather', 'allreduce', 'alltoall', 'reduce_scatter'])
def test_six_operators_have_complete_hierarchical_interfaces(tmp_path, operator):
    from vericcl.planner.group_search import enumerate_hierarchy_plans
    from vericcl.solver.demands import build_solver_problem
    inputs, topology = setup(tmp_path, operator)
    spec = next(item for item in enumerate_hierarchy_plans(inputs, topology) if item.rail_assignments)
    assert spec.plan.collective == inputs.collective
    assert spec.plan.final_outputs.values
    assert len({node.stage_id for node in spec.plan.nodes}) >= 3
    for node in spec.plan.nodes:
        problem = build_solver_problem(node, inputs, topology)
        for demand in problem.demands:
            participants = set().union(*(set(node.participants_by_slice[s]) for s in demand.member_slice_ids))
            assert all(set(path) <= participants <= set(node.communication_group) for path in demand.candidate_paths)


def test_asymmetric_gateways_and_zero_rail_shares_are_candidates(tmp_path):
    from vericcl.planner.group_search import discover_rails, integer_allocations
    inputs, topology = setup(tmp_path, 'allgather')
    # Only rank 2 has inter-node links on the second node; rank 0 and 1
    # are both gateways on the first node. No equal-count assumption.
    links = {key: edge for key, edge in topology.links.items()
             if topology.node_membership[key.src_rank] == topology.node_membership[key.dst_rank]
             or 3 not in (key.src_rank, key.dst_rank)}
    asymmetric = replace(topology, links=links, gateways=frozenset({0, 1, 2}), isomorphism_signature='')
    assert discover_rails(asymmetric) == ((0, 2), (1, 2))
    allocations = tuple(integer_allocations(3, 2))
    assert (0, 3) in allocations and (3, 0) in allocations and (1, 2) in allocations


def test_explicit_stage_count_is_a_hard_constraint(tmp_path):
    from vericcl.planner.group_search import enumerate_hierarchy_plans
    inputs, topology = setup(tmp_path, 'allreduce', stages=4)
    candidates = list(itertools.islice(enumerate_hierarchy_plans(inputs, topology), 8))
    assert candidates
    assert all({node.stage_id for node in candidate.plan.nodes} == set(range(4)) for candidate in candidates)


def test_per_chunk_roles_and_group_boundary_filter_paths(tmp_path):
    from vericcl.planner.model import PlanNode, StageInterface
    from vericcl.semantics.collective import CollectiveKind, CollectiveSpec, OutputSlot
    from vericcl.solver.demands import build_solver_problem
    inputs, topology = setup(tmp_path, 'broadcast')
    group = (0, 1, 2)
    node = PlanNode('roles', 0, CollectiveSpec(CollectiveKind.BROADCAST, 'float32', None, 0, False),
        group, StageInterface({OutputSlot(0, 0): frozenset({0})}),
        StageInterface({OutputSlot(2, 0): frozenset({0})}),
        frozenset(key for key in topology.links if key.src_rank in group and key.dst_rank in group), frozenset(),
        participants_by_slice={0: (0, 2)})
    demand = build_solver_problem(node, inputs, topology).demands[0]
    assert demand.candidate_paths == ((0, 2),)


def test_reuse_preserves_semantics_and_prediction_with_fewer_real_models(tmp_path):
    from vericcl.execution.budget import StandaloneLease
    from vericcl.planner.group_search import HierarchyContext, plan_hierarchy
    from vericcl.verification.semantics import check_semantics
    inputs, topology = setup(tmp_path, 'broadcast', 2, stages=1)
    streams = [plan_hierarchy(inputs, topology, None, HierarchyContext(StandaloneLease.after(30), reuse_subproblems=reuse))
               for reuse in (True, False)]
    candidates = [next(iter(stream)) for stream in streams]
    assert all(check_semantics(candidate.schedule, inputs).status.value == 'valid' for candidate in candidates)
    assert candidates[0].communication_time_us == pytest.approx(candidates[1].communication_time_us)
    assert streams[0].model_calls < streams[1].model_calls
    assert streams[0].model_calls > 0


def test_fine_phase_uses_joint_route_and_timing_models(tmp_path):
    from vericcl.execution.budget import StandaloneLease
    from vericcl.planner.group_search import HierarchyContext, plan_hierarchy
    inputs, topology = setup(tmp_path, 'broadcast', 2, stages=1)
    stream = plan_hierarchy(inputs, topology, None, HierarchyContext(StandaloneLease.after(20)))
    candidates = list(stream)
    assert any(candidate.stage == 'fine' for candidate in candidates)
    assert stream.fine_model_calls > 0


def test_manual_entry_translates_explicit_state_interfaces(tmp_path):
    from vericcl.planner.group_search import enumerate_hierarchy_plans
    docs = request_documents('broadcast', 2)
    docs['sketch']['hyperparameters'].update(total_size_bytes=1024)
    docs['atom']['manual'] = {'enabled': True, 'nodes': [{'id': 'broadcast', 'stage_id': 0,
        'operator': 'broadcast', 'group': [0, 1], 'root': 0,
        'input_states': [{'rank': 0, 'offset': 0, 'contributors': [0]}],
        'output_states': [{'rank': 0, 'offset': 0, 'contributors': [0]},
                          {'rank': 1, 'offset': 0, 'contributors': [0]}]}]}
    inputs = load_request(write_request(tmp_path, docs))
    plans = tuple(enumerate_hierarchy_plans(inputs, load_topology(inputs)))
    assert len(plans) == 1 and plans[0].plan.planning_reason == 'manual_hierarchy'


@pytest.mark.parametrize('operator', ['broadcast', 'reduce', 'allgather', 'allreduce', 'alltoall', 'reduce_scatter'])
def test_hierarchical_coarse_and_fine_schedules_preserve_full_semantics(tmp_path, operator):
    from vericcl.execution.budget import StandaloneLease
    from vericcl.planner.group_search import HierarchyContext, plan_hierarchy
    from vericcl.verification.semantics import check_semantics
    inputs, topology = setup(tmp_path, operator, stages=4 if operator == 'allreduce' else 3)
    stream = plan_hierarchy(inputs, topology, None, HierarchyContext(StandaloneLease.after(40), coarse_batch_size=1))
    iterator = iter(stream)
    candidates = list(itertools.islice(iterator, 2))
    iterator.close()
    assert {candidate.stage for candidate in candidates} == {'coarse', 'fine'}
    for candidate in candidates:
        checked = check_semantics(candidate.schedule, inputs)
        assert checked.status.value == 'valid', (checked.code, checked.message, checked.evidence)


def test_current_model_changes_representative_link_costs_without_changing_structure(tmp_path):
    from vericcl.performance.model import CostModel, ServiceCost
    from vericcl.performance.topology import project_topology, link_cost_key
    inputs, topology = setup(tmp_path, 'broadcast', 2)
    model = CostModel('updated', {link_cost_key(0, 1): (ServiceCost(5, .25),),
                                 link_cost_key(1, 0): (ServiceCost(2, .1),)})
    projected = project_topology(topology, model, 1024)
    assert projected.links.keys() == topology.links.keys()
    from vericcl.topology.model import LinkKey
    assert projected.link(LinkKey(0, 1)).performance.beta_us_per_byte == .25
    assert topology.link(LinkKey(0, 1)).performance.beta_us_per_byte == .01
