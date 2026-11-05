from dataclasses import replace

from tests.unit.input.v5_support import request_documents, write_request
from vericcl.input.loader import load_request
from vericcl.topology import load_topology


def test_hierarchy_defaults_keep_optional_restrictions_disabled(tmp_path):
    from vericcl.planner.group_search import hierarchy_options
    inputs = load_request(write_request(tmp_path))
    options = hierarchy_options(inputs)
    assert not options.symmetry and not options.shortest_paths and not options.batching
    assert options.batch_slices == 1
    assert options.reuse_subproblems


def test_symmetry_does_not_disable_equivalent_subproblem_reuse(tmp_path):
    from vericcl.planner.group_search import hierarchy_options
    from vericcl.planner.direct import build_direct_plan
    from vericcl.solver.demands import build_solver_problem
    from vericcl.solver.templates import build_solver_templates
    docs = request_documents('broadcast', 2)
    docs['atom']['strategies'] = {'symmetry': True}
    inputs = load_request(write_request(tmp_path, docs))
    topology = load_topology(inputs)
    plan = build_direct_plan(inputs, topology)
    problems = tuple(build_solver_problem(node, inputs, topology) for node in plan.nodes)
    assert hierarchy_options(inputs).reuse_subproblems
    assert len(build_solver_templates(problems, plan.planning_mode)) == 1
    separate = build_solver_templates(problems, plan.planning_mode, reuse=False)
    assert len(separate) == inputs.hyperparameters.slice_count


def test_manual_residue_is_ignored_unless_enabled(tmp_path):
    from vericcl.planner.group_search import enumerate_hierarchy_plans
    inputs = load_request(write_request(tmp_path))
    changed = replace(inputs, strategies=replace(inputs.strategies, manual_hierarchy=({'invalid': True},)))
    assert next(enumerate_hierarchy_plans(changed, load_topology(changed))).plan.planning_reason != 'manual_hierarchy'


def test_batch_grouping_is_independent_of_channels_and_preserves_slices(tmp_path):
    from vericcl.planner.group_search import enumerate_hierarchy_plans
    docs = request_documents('broadcast', 2)
    docs['atom']['strategies'] = {'batching': True}
    docs['atom']['stage_num'] = 1
    inputs = load_request(write_request(tmp_path, docs))
    plans = [next(enumerate_hierarchy_plans(replace(inputs, solver=replace(inputs.solver, max_channels=channels)), load_topology(inputs))).plan
             for channels in (1, 3)]
    assert len(plans[0].nodes) == len(plans[1].nodes) == 1
    assert len(plans[0].nodes[0].logical_input.values) == 4
    assert plans[0].slice_count == 4


def test_reference_slice_ban_follows_original_bytes_after_parameter_change(tmp_path):
    from vericcl.planner.direct import build_direct_plan
    from vericcl.solver.demands import build_solver_problem
    docs = request_documents('broadcast', 3)
    docs['sketch']['hyperparameters'].update(slice_size_bytes=1024, reference_slice_size_bytes=512)
    docs['atom']['forbidden_transfers'] = [[1, 0, 1, 0]]
    inputs = load_request(write_request(tmp_path, docs))
    topology = load_topology(inputs)
    node = build_direct_plan(inputs, topology).nodes[0]
    problem = build_solver_problem(node, inputs, topology)
    target = next(d for d in problem.demands if d.required_leaf_rank == 1)
    assert target.candidate_paths == ((0, 2, 1),)


def test_explicit_topology_groups_are_used_as_candidate_partitions(tmp_path):
    from vericcl.planner.group_search import candidate_partitions
    docs = request_documents('allreduce', 4)
    docs['topology']['groups'] = [[0, 2], [1, 3]]
    inputs = load_request(write_request(tmp_path, docs))
    assert ((0, 2), (1, 3)) in candidate_partitions(inputs, load_topology(inputs))


def test_optional_symmetry_is_applied_inside_native_representative_model(tmp_path):
    from vericcl.solver.milp import solve_milp
    from vericcl.solver.demands import build_solver_problem
    from vericcl.solver.budget import ModelBudget
    from vericcl.planner.direct import build_direct_plan
    from vericcl.input.models import ObjectiveMode
    import time
    docs = request_documents('broadcast', 2)
    docs['atom']['strategies'] = {'symmetry': True}
    inputs = load_request(write_request(tmp_path, docs))
    topology = load_topology(inputs)
    problem = build_solver_problem(build_direct_plan(inputs, topology).nodes[0], inputs, topology)
    now = time.monotonic()
    candidate = solve_milp(problem, 2, ObjectiveMode.LATENCY, ModelBudget(5, now, now + 5), None)
    schedule = candidate.node_schedules[problem.node.node_id]
    assert schedule.metadata['symmetry_constraints'] > 0
    assert {transfer.channel for transfer in schedule.transfers} == {0}


def test_batching_enabled_has_a_real_joint_model_and_valid_schedule(tmp_path):
    from vericcl.planner.group_search import HierarchyContext, plan_hierarchy
    from vericcl.execution.budget import StandaloneLease
    from vericcl.verification.semantics import check_semantics
    docs = request_documents('broadcast', 2)
    docs['atom']['strategies'] = {'batching': True}
    docs['atom']['stage_num'] = 1
    docs['sketch']['solver']['max_channels'] = 2
    inputs = load_request(write_request(tmp_path, docs))
    stream = plan_hierarchy(inputs, load_topology(inputs), None, HierarchyContext(StandaloneLease.after(20)))
    candidates = list(stream)
    fine = next(candidate for candidate in candidates if candidate.stage == 'fine')
    assert check_semantics(fine.schedule, inputs).status.value == 'valid'
    assert stream.fine_model_calls == 1
