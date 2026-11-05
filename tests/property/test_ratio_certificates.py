"""Proof checking validates wire types while preserving exact legacy integer pairs."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from tests.property.test_search_certificates import space_fixture
from vericcl.performance.model import CostModel, ServiceCost
from vericcl.verification import proof_check
from vericcl.xml.lower import lower_and_check


@pytest.mark.parametrize('where', ['incumbent_ratio', 'leaf'])
@pytest.mark.parametrize('style', ['float', 'boolean'])
def test_certificate_rejects_noninteger_ratio_components(tmp_path, where, style):
    space = space_fixture(tmp_path)
    model = CostModel('integer-proof', {key: (ServiceCost(0, 1),) for key in space.context.model.services},
                       space.context.model.resources)
    context = replace(space.context, model=model)
    bundle = lower_and_check(space.bundle.source, context)
    assert bundle.can_execute and bundle.prediction.exact_completion_ratio[1] == 1
    space = replace(space, bundle=bundle, context=context)
    certificate = proof_check.certify_space(space, use_bound=False)
    target = certificate if where == 'incumbent_ratio' else certificate['leaves'][0]
    key = 'incumbent_ratio' if where == 'incumbent_ratio' else 't_pred_ratio'
    if style == 'float':
        target[key] = list(map(float, target[key]))
    else:
        target[key][1] = True
    assert proof_check.check_no_improvement(certificate, space=space).status == 'invalid_certificate'


def test_legacy_large_integer_pair_is_accepted_by_exact_value(tmp_path, monkeypatch):
    space = space_fixture(tmp_path)
    ratio = (10**620 + 1, 10**620 + 3)
    baseline, replay = proof_check._baseline, proof_check.replay_decision

    def altered(bundle):
        # Controlled wire-format fixture; hardware and model evidence are not asserted here.
        return replace(bundle, prediction=replace(bundle.prediction, exact_completion_ratio=ratio))

    monkeypatch.setattr(proof_check, '_baseline', lambda value: altered(baseline(value)))
    monkeypatch.setattr(proof_check, 'replay_decision', lambda value, decision: altered(replay(value, decision)))
    certificate = deepcopy(proof_check.certify_space(space, use_bound=False))
    certificate['incumbent_ratio'] = list(ratio)
    for leaf in certificate['leaves']:
        leaf['t_pred_ratio'] = list(ratio)
    saved = json.loads(json.dumps(certificate))
    assert proof_check.check_no_improvement(saved, space=space).status == 'proven_no_improvement'
