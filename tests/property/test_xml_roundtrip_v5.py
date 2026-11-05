import copy
import hashlib

import pytest
from lxml import etree

from tests.integration.test_lowering_v5 import candidate_fixture


def test_v5_xml_sidecar_roundtrip_preserves_physical_work_and_contributions(tmp_path):
    from vericcl.xml.lower import lower_and_check
    from vericcl.xml.parser import check_xml_bundle
    candidate, context = candidate_fixture(tmp_path, 'allreduce')
    original = lower_and_check(candidate, context)
    restored = check_xml_bundle(original.xml_text, original.sidecar, context)
    assert restored.can_execute
    assert restored.version_schedule == original.version_schedule
    assert restored.executable == original.executable
    assert restored.prediction.t_pred_us == original.prediction.t_pred_us


def test_xml_hash_mismatch_is_rejected_before_execution(tmp_path):
    from vericcl.xml.lower import lower_and_check
    from vericcl.xml.parser import check_xml_bundle
    candidate, context = candidate_fixture(tmp_path, 'broadcast')
    original = lower_and_check(candidate, context)
    changed = original.xml_text.replace('cnt="1"', 'cnt="2"', 1)
    restored = check_xml_bundle(changed, original.sidecar, context)
    assert not restored.can_execute
    assert restored.checks['xml'].status.value == 'invalid'


@pytest.mark.parametrize('fault', ['offset', 'self_dependency', 'contribution'])
def test_corrupt_xml_or_semantic_mapping_cannot_pass_with_updated_hash(tmp_path, fault):
    from vericcl.xml.lower import lower_and_check
    from vericcl.xml.parser import check_xml_bundle
    candidate, context = candidate_fixture(tmp_path, 'allreduce')
    original = lower_and_check(candidate, context)
    root, sidecar = etree.fromstring(original.xml_text.encode()), copy.deepcopy(original.sidecar)
    if fault == 'offset':
        root.find('.//step[@type="s"]').set('srcoff', '32768')
    elif fault == 'self_dependency':
        gpu = root.find('gpu')
        tb = gpu.find('tb')
        step = tb.find('step')
        step.set('depid', tb.get('id')); step.set('deps', step.get('s')); step.set('hasdep', '1')
    else:
        sidecar['version_schedule']['outputs'][0]['state_id'] = sidecar['version_schedule']['initial_states'][0]['state_id']
    xml = etree.tostring(root, encoding='unicode') + '\n'
    sidecar['xml_sha256'] = hashlib.sha256(xml.encode()).hexdigest()
    restored = check_xml_bundle(xml, sidecar, context)
    assert not restored.can_execute
    assert any(result.status.value == 'invalid' for result in restored.checks.values())


@pytest.mark.parametrize('fault', ['chunk_geometry', 'input_geometry', 'channel_limit'])
def test_matching_sidecar_cannot_override_runtime_request_geometry(tmp_path, fault):
    from vericcl.xml.lower import lower_and_check
    from vericcl.xml.parser import check_xml_bundle
    from vericcl.xml.v5_lower import _deserialize_program
    from vericcl.xml.v5_program import emit_native_xml
    candidate, context = candidate_fixture(tmp_path, 'allreduce')
    original = lower_and_check(candidate, context)
    sidecar = copy.deepcopy(original.sidecar)
    if fault == 'chunk_geometry':
        sidecar['program']['chunks_per_loop'] *= 2
    elif fault == 'input_geometry':
        for name in ('program', 'storage'):
            sidecar[name]['input_slots'][0] *= 2
    else:
        sidecar['program']['channels'] = 2
        for block in sidecar['program']['blocks']:
            if block['send_peer'] >= 0 or block['recv_peer'] >= 0:
                block['channel'] = 1
        for instruction in sidecar['storage']['instructions']:
            if instruction['kind'] in {'send', 'rrc'}:
                instruction['channel'] = 1
    xml = emit_native_xml(_deserialize_program(sidecar['program']), context.inputs)
    sidecar['xml_sha256'] = hashlib.sha256(xml.encode()).hexdigest()
    checked = check_xml_bundle(xml, sidecar, context)
    assert not checked.can_execute
    assert checked.checks['launch'].status.value == 'invalid'


def test_sidecar_is_strict_finite_json_and_reloads_after_serialization(tmp_path):
    import json
    from vericcl.xml.lower import lower_and_check
    from vericcl.xml.parser import check_xml_bundle
    candidate, context = candidate_fixture(tmp_path, 'broadcast')
    bundle = lower_and_check(candidate, context)
    encoded = json.dumps(bundle.sidecar, allow_nan=False)
    restored = check_xml_bundle(bundle.xml_text, json.loads(encoded), context)
    assert restored.can_execute


def test_matching_xml_and_sidecar_self_deadlock_is_still_rejected(tmp_path):
    from vericcl.xml.lower import lower_and_check
    from vericcl.xml.parser import check_xml_bundle
    from vericcl.xml.v5_lower import _deserialize_program
    from vericcl.xml.v5_program import emit_native_xml
    candidate, context = candidate_fixture(tmp_path, 'allreduce')
    bundle = lower_and_check(candidate, context)
    sidecar = copy.deepcopy(bundle.sidecar)
    block = sidecar['program']['blocks'][0]
    step = block['steps'][0]
    step.update(dependency_tb=block['tb_id'], dependency_step=0, has_dependence=True)
    xml = emit_native_xml(_deserialize_program(sidecar['program']), context.inputs)
    sidecar['xml_sha256'] = hashlib.sha256(xml.encode()).hexdigest()
    checked = check_xml_bundle(xml, sidecar, context)
    assert not checked.can_execute
    assert checked.checks['deadlock'].status.value == 'invalid'


def test_same_xml_is_reevaluated_under_the_new_model_without_rewriting_inputs(tmp_path):
    from dataclasses import replace
    from vericcl.performance.model import ServiceCost
    from vericcl.xml.lower import lower_and_check
    from vericcl.xml.parser import check_xml_bundle
    candidate, context = candidate_fixture(tmp_path, 'allreduce')
    original = lower_and_check(candidate, context)
    model, differences = context.model.with_updates('recalibrated',
        services={key: (ServiceCost(costs[0].alpha_us * 2, costs[0].beta_us_per_byte * 2),)
                  for key, costs in context.model.services.items()},
        resources={key: capacity / 2 for key, capacity in context.model.resources.items()})
    updated = check_xml_bundle(original.xml_text, original.sidecar, replace(context, model=model))
    assert updated.can_execute
    assert updated.xml_sha256 == original.xml_sha256
    assert updated.prediction.t_pred_us == pytest.approx(original.prediction.t_pred_us * 2)
    assert original.sidecar['model_version'] == 'lowering-fixture'
    assert updated.prediction.model_version == 'recalibrated'
