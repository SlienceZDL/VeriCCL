from dataclasses import replace
from copy import deepcopy

from tests.integration.test_lowering_v5 import candidate_fixture
from vericcl.xml.lower import lower_and_check


def space_fixture(tmp_path, *, channels=False):
    from vericcl.verification.proof_check import NativeDecisionSpace
    candidate, context = candidate_fixture(tmp_path, 'broadcast', ranks=2, channels=2 if channels else 1)
    bundle = lower_and_check(candidate, context)
    return NativeDecisionSpace(bundle, context, vary_order=True, vary_channels=channels)


def test_complete_small_space_is_replayed_with_exact_model(tmp_path):
    from vericcl.verification.proof_check import certify_space, check_no_improvement
    space = space_fixture(tmp_path)
    certificate = certify_space(space, use_bound=False)
    assert certificate['leaves']
    result = check_no_improvement(certificate, space=space)
    assert result.status == 'proven_no_improvement', result
    assert result.checked_leaves == len(certificate['leaves'])
    assert set(result.families) == {'order'}
    assert result.scope_id == space.scope_id
    assert not result.full_repair_space


def test_missing_branch_and_duplicate_leaf_are_rejected(tmp_path):
    from vericcl.verification.proof_check import certify_space, check_no_improvement
    space = space_fixture(tmp_path)
    certificate = certify_space(space, use_bound=False)
    incomplete = deepcopy(certificate); incomplete['leaves'].pop()
    assert check_no_improvement(incomplete, space=space).status == 'incomplete'
    duplicate = deepcopy(certificate); duplicate['leaves'].append(duplicate['leaves'][0])
    assert check_no_improvement(duplicate, space=space).status == 'invalid_certificate'


def test_model_identity_and_actual_parameters_both_bind_proof(tmp_path):
    from vericcl.performance.model import ServiceCost, CostModel
    from vericcl.verification.proof_check import certify_space, check_no_improvement
    space = space_fixture(tmp_path)
    certificate = certify_space(space, use_bound=False)
    new, _ = space.context.model.with_updates('new')
    changed = replace(space, context=replace(space.context, model=new))
    assert check_no_improvement(certificate, space=changed).status == 'stale_certificate'
    same_name = CostModel(space.context.model.version,
                         {key: tuple(replace(c, alpha_us=c.alpha_us+1) for c in values)
                          for key, values in space.context.model.services.items()}, space.context.model.resources)
    changed = replace(space, context=replace(space.context, model=same_name))
    assert check_no_improvement(certificate, space=changed).status == 'stale_certificate'


def test_forged_lower_bound_and_unverifiable_dual_cannot_close_space(tmp_path):
    from vericcl.verification.proof_check import certify_space, check_no_improvement, optimistic_bound
    space = space_fixture(tmp_path)
    certificate = certify_space(space, use_bound=False)
    bound = optimistic_bound(space)
    forged = deepcopy(certificate)
    forged['leaves'] = []
    forged['bound'] = deepcopy(bound.certificate)
    forged['bound']['rhs'] = ['999999'] * len(forged['bound']['rhs'])
    assert check_no_improvement(forged, space=space).status == 'invalid_certificate'
    forged['bound'] = deepcopy(bound.certificate)
    forged['bound']['dual'] = ['2'] * len(forged['bound']['dual'])
    assert check_no_improvement(forged, space=space).status == 'invalid_certificate'


def test_fake_times_and_numerical_solver_status_are_not_exact_evidence(tmp_path):
    from vericcl.verification.proof_check import certify_space, check_no_improvement
    space = space_fixture(tmp_path)
    certificate = certify_space(space, use_bound=False)
    wrong = deepcopy(certificate)
    wrong['leaves'][0]['t_pred_ratio'] = [999999, 1]
    assert check_no_improvement(wrong, space=space).status == 'invalid_certificate'
    wrong = deepcopy(certificate)
    wrong['leaves'][0] = {'key': wrong['leaves'][0]['key'], 'status': 'SCIP_OPTIMAL', 'dual_bound': 1e9}
    assert check_no_improvement(wrong, space=space).status == 'invalid_certificate'


def test_scoped_proof_does_not_close_unexpanded_path_or_structural_branches(tmp_path):
    from vericcl.verification.proof_check import certify_space, check_no_improvement
    from vericcl.verification.search_coverage import SearchCoverage
    space = space_fixture(tmp_path)
    proof = check_no_improvement(certify_space(space, use_bound=False), space=space)
    coverage = SearchCoverage('outer', space.context.model, ('order', 'path', 'structure'))
    coverage.declare_scope('order', space.scope_id)
    coverage.close_branch('order', proof)
    result = coverage.snapshot()
    assert not result['proven_no_improvement']
    assert result['unresolved_branches'] == ['path', 'structure']


def test_partial_budget_is_never_reported_as_no_improvement(tmp_path):
    from vericcl.verification.proof_check import certify_space, check_no_improvement
    space = space_fixture(tmp_path)
    certificate = certify_space(space, use_bound=False)
    space.context.lease.cancel()
    assert check_no_improvement(certificate, space=space).status == 'budget_exhausted'


def test_sound_reconstructed_root_bound_can_close_serial_work(tmp_path):
    from vericcl.verification.proof_check import NativeDecisionSpace, certify_space, check_no_improvement
    from vericcl.xml.v5_program import lower_program, emit_native_xml
    from vericcl.xml.v5_lower import evaluate_program
    space = space_fixture(tmp_path)
    previous, operations = None, []
    for op in space.bundle.storage_plan.instructions:
        operations.append(replace(op, dependencies=op.dependencies | ({previous} if previous else set())))
        previous = op.operation_id
    storage = replace(space.bundle.storage_plan, instructions=tuple(operations))
    program = lower_program(storage, space.context)
    bundle = evaluate_program(space.bundle.source, storage, program, emit_native_xml(program, space.context.inputs), space.context)
    assert bundle.can_execute
    space = NativeDecisionSpace(bundle, space.context)
    certificate = certify_space(space)
    assert certificate['bound'] and not certificate['leaves']
    assert check_no_improvement(certificate, space=space).status == 'proven_no_improvement'


def test_improving_channel_assignment_is_returned_as_counterexample(tmp_path):
    from vericcl.verification.proof_check import NativeDecisionSpace, certify_space, check_no_improvement
    from vericcl.verification.bottlenecks import mandatory_dependencies
    from vericcl.xml.v5_program import lower_program, emit_native_xml
    from vericcl.xml.v5_lower import evaluate_program
    original = space_fixture(tmp_path, channels=True)
    storage = original.bundle.storage_plan
    mandatory = mandatory_dependencies(storage)
    storage = replace(storage, instructions=tuple(replace(op, channel=0, dependencies=mandatory[op.operation_id])
                                                 for op in storage.instructions))
    program = lower_program(storage, original.context)
    bundle = evaluate_program(original.bundle.source, storage, program, emit_native_xml(program, original.context.inputs), original.context)
    assert bundle.can_execute
    space = NativeDecisionSpace(bundle, original.context, vary_channels=True)
    certificate = certify_space(space, use_bound=False)
    result = check_no_improvement(certificate, space=space)
    assert result.status == 'improvement_found'
    assert result.counterexample and result.checked_leaves


def test_relaxation_is_below_every_replayed_decision_with_measured_profiles(tmp_path):
    from fractions import Fraction
    from vericcl.performance.profiles import RateProfile
    from vericcl.verification.proof_check import optimistic_bound, decisions, replay_decision
    space = space_fixture(tmp_path, channels=True)
    profile = RateProfile('measured-pair', ('send:0:1:c0', 'send:0:1:c1'),
                          {'send:0:1:c0': 50, 'send:0:1:c1': 50}, resources=('link:0:1',),
                          startup_us={'send:0:1:c0': 0, 'send:0:1:c1': 0}, measured_at='2026-11-03T00:00:00Z')
    model, _ = space.context.model.with_updates('profiles', profiles=(profile,))
    space = replace(space, context=replace(space.context, model=model))
    bound = optimistic_bound(space)
    for decision in decisions(space):
        bundle = replay_decision(space, decision)
        if bundle.can_execute:
            assert bound.value <= Fraction(*bundle.prediction.exact_completion_ratio)
