import pytest

from vericcl.errors import InputValidationError
from vericcl.input.loader import load_request
from vericcl.semantics.ranges import DataRange, bind_constraints
from tests.unit.input.v5_support import request_documents, write_request


def documents(enabled=True):
    docs = request_documents()
    docs['atom']['manual'] = {'enabled': enabled, 'nodes': [{
        'id': 'exchange', 'stage_id': 0, 'operator': 'allgather', 'group': [0, 1, 2, 3],
        'atoms': [{'slice_ids': [0, 1], 'src_ranks': [0], 'dst_ranks': [1, 3]}],
    }]}
    return docs


def test_manual_groups_expand_before_solving_and_only_ban_directed_edges(tmp_path):
    inputs = load_request(write_request(tmp_path, documents()))
    constraints = bind_constraints(inputs)
    assert {(b.data.start, b.src_rank, b.dst_rank, b.stage_id) for b in constraints.bans} == {
        (0, 0, 1, 0), (0, 0, 3, 0), (1024, 0, 1, 0), (1024, 0, 3, 0)}
    contribution = [DataRange(0, 0, 1024)]
    assert not constraints.allows(contribution, 0, 1, 0)
    assert constraints.allows(contribution, 0, 2, 0)
    assert constraints.allows(contribution, 2, 1, 0)
    assert constraints.allows(contribution, 1, 0, 0)
    assert constraints.allows(contribution, 0, 1, 1)
    assert inputs.hyperparameters.reference_slice_size_bytes == 1024


def test_disabled_manual_atoms_do_not_constrain_automatic_synthesis(tmp_path):
    inputs = load_request(write_request(tmp_path, documents(False)))
    assert inputs.atom_constraints.forbidden_transfers == ()
    assert inputs.hyperparameters.reference_slice_size_bytes is None


def test_manual_atoms_and_legacy_bans_merge_without_duplicate_constraints(tmp_path):
    docs = documents()
    docs['atom']['forbidden_transfers'] = [[0, 0, 1, 0], [2, 1, 0, 0]]
    inputs = load_request(write_request(tmp_path, docs))
    assert len(inputs.atom_constraints.forbidden_transfers) == 5


@pytest.mark.parametrize('field,value', [('time', 0), ('size', 1024), ('path', [0, 1]), ('stage_id', 1)])
def test_atom_group_accepts_only_variable_domains(tmp_path, field, value):
    docs = documents()
    docs['atom']['manual']['nodes'][0]['atoms'][0][field] = value
    with pytest.raises(InputValidationError, match='unknown'):
        load_request(write_request(tmp_path, docs))


@pytest.mark.parametrize('field,value', [('slice_ids', [16]), ('src_ranks', [4]),
    ('dst_ranks', [0]), ('slice_ids', []), ('slice_ids', [-1]), ('src_ranks', [True])])
def test_manual_atom_invalid_domains_fail_before_solver(tmp_path, field, value):
    docs = documents()
    docs['atom']['manual']['nodes'][0]['atoms'][0][field] = value
    with pytest.raises(InputValidationError):
        load_request(write_request(tmp_path, docs))


def test_manual_atom_ranks_stay_inside_stage_group(tmp_path):
    docs = documents()
    docs['atom']['manual']['nodes'][0]['group'] = [0, 1, 2]
    with pytest.raises(InputValidationError, match='group'):
        load_request(write_request(tmp_path, docs))


def test_manual_atom_stage_uses_fixed_stage_domain(tmp_path):
    docs = documents()
    docs['atom']['stage_num'] = 1
    docs['atom']['manual']['nodes'][0]['stage_id'] = 1
    with pytest.raises(InputValidationError, match='stage'):
        load_request(write_request(tmp_path, docs))


def test_reduction_payload_and_repartition_inherit_manual_ban(tmp_path):
    inputs = load_request(write_request(tmp_path, documents()))
    constraints = bind_constraints(inputs)
    assert not constraints.allows([DataRange(0, 512, 2048), DataRange(1, 512, 2048)], 0, 1, 0)
    assert constraints.allows([DataRange(0, 2048, 4096), DataRange(1, 2048, 4096)], 0, 1, 0)


def test_manual_plan_solver_demands_retain_alternative_route(tmp_path):
    from vericcl.planner.manual import plan_manual
    from vericcl.solver.demands import build_solver_problem
    from vericcl.topology import load_topology
    docs = request_documents('allgather', 3)
    docs['sketch']['hyperparameters']['total_size_bytes'] = 1024
    docs['atom']['manual'] = {'enabled': True, 'nodes': [{
        'id': 'exchange', 'stage_id': 0, 'operator': 'allgather', 'group': [0, 1, 2],
        'input_states': [{'rank': r, 'offset': 0, 'contributors': [r]} for r in range(3)],
        'output_states': [{'rank': r, 'offset': s, 'contributors': [s]} for r in range(3) for s in range(3)],
        'atoms': [{'slice_ids': [0], 'src_ranks': [0], 'dst_ranks': [1]}],
    }]}
    inputs = load_request(write_request(tmp_path, docs))
    topology = load_topology(inputs)
    plan = plan_manual(inputs, topology)
    problem = build_solver_problem(plan.nodes[0], inputs, topology)
    demand = next(item for item in problem.demands if set(item.member_slice_ids) == {0} and item.required_leaf_rank == 1)
    assert demand.candidate_paths == ((0, 2, 1),)
