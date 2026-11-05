from dataclasses import replace

from tests.integration.test_lowering_v5 import candidate_fixture
from vericcl.performance.model import ServiceCost
from vericcl.performance.simulation import predict
from vericcl.verification.online.model_publication import ComponentUpdates, publish_model
from vericcl.verification.proof_check import NativeDecisionSpace, certify_space, check_no_improvement
from vericcl.xml.lower import lower_and_check
from vericcl.xml.parser import check_xml_bundle


def test_same_xml_uses_new_costs_and_cannot_reuse_old_model_certificate(tmp_path):
    candidate, context = candidate_fixture(tmp_path, 'broadcast')
    old = lower_and_check(candidate, context)
    assert old.can_execute
    certificate = certify_space(NativeDecisionSpace(old, context), use_bound=False)
    assert check_no_improvement(certificate, space=NativeDecisionSpace(old, context)).status == 'proven_no_improvement'
    services = {key: (ServiceCost(value[0].alpha_us * 2, value[0].beta_us_per_byte * 2,
                                 'measured', 1, None, '2026-11-04T00:00:00Z'),)
                for key, value in context.model.services.items()}
    new_model = publish_model(context.model, ComponentUpdates(services=services,
        signatures={'service:' + key: 'fixture-runtime' for key in services}), initial=context.model)
    new_context = replace(context, model=new_model)
    reloaded = check_xml_bundle(old.xml_text, old.sidecar, new_context)
    assert reloaded.can_execute and reloaded.xml_sha256 == old.xml_sha256
    assert reloaded.prediction.model_version == new_model.version
    assert reloaded.prediction.t_pred_us > old.prediction.t_pred_us
    proof = check_no_improvement(certificate, space=NativeDecisionSpace(reloaded, new_context))
    assert proof.status != 'proven_no_improvement'
    assert predict(old.executable, new_model, exact=True).t_pred_us == reloaded.prediction.t_pred_us
