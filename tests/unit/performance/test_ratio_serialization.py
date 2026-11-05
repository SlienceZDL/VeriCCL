"""Large exact completion certificates survive ordinary bounded JSON readers."""
from dataclasses import replace
import json
import sys

import pytest

from vericcl.performance.events import Executable, WorkOperation
from vericcl.performance.simulation import predict
from vericcl.workflow_state import plain
from tests.unit.performance.test_event_progress import model


def prediction():
    return predict(Executable((WorkOperation('copy', 'copy', 'wire', 1),)),
                   model(alpha=.1, beta=.2), exact=True)


def test_small_prediction_ratio_keeps_the_existing_json_shape():
    assert json.loads(json.dumps(plain(prediction())))['exact_completion_ratio'] == [3, 10]


def test_large_prediction_ratio_survives_the_minimum_decimal_limit():
    ratio = (10**5000 + 1, 10**5000 + 3)
    value = replace(prediction(), exact_completion_ratio=ratio)
    before = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(640)
        saved = json.loads(json.dumps(plain(value)))
        from vericcl.performance.ratios import ratio_from_json
        assert ratio_from_json(saved['exact_completion_ratio']) == ratio
        assert sys.get_int_max_str_digits() == 640
    finally:
        sys.set_int_max_str_digits(before)


@pytest.mark.parametrize('value', [
    [True, 1], [1, 0], [1, -2], ['1', 2], [1],
    {'encoding': 'unknown', 'numerator': '0x1', 'denominator': '0x2'},
    {'encoding': 'hex_integer_ratio_v1', 'numerator': '0x01', 'denominator': '0x2'},
    {'encoding': 'hex_integer_ratio_v1', 'numerator': '0x1', 'denominator': '0x0'},
    {'encoding': 'hex_integer_ratio_v1', 'numerator': '0x1', 'denominator': '0x2', 'extra': 0},
])
def test_invalid_ratio_encodings_are_rejected(value):
    from vericcl.performance.ratios import ratio_from_json
    with pytest.raises(ValueError):
        ratio_from_json(value)


def test_exact_proof_leaf_uses_the_same_lossless_ratio_encoding(monkeypatch):
    from types import SimpleNamespace
    from vericcl.verification import proof_check
    ratio = (10**5000 + 1, 10**5000 + 3)
    bundle = SimpleNamespace(can_execute=True, xml_sha256='checked',
                             prediction=replace(prediction(), exact_completion_ratio=ratio))
    monkeypatch.setattr(proof_check, 'replay_decision', lambda *_: bundle)
    record = proof_check._record(None, {'key': 'leaf'})
    saved = json.loads(json.dumps(record))
    from vericcl.performance.ratios import ratio_from_json
    assert ratio_from_json(saved['t_pred_ratio']) == ratio
