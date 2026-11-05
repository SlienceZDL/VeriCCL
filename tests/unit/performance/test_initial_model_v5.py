from dataclasses import replace

import pytest

from tests.unit.input.v5_support import request_documents, write_request
from vericcl.errors import InputValidationError
from vericcl.input.loader import load_request
from vericcl.performance.initial import initial_cost_model
from vericcl.performance.serialization import model_from_json
from vericcl.verification.online.model_publication import model_json


def test_explicit_link_and_local_costs_cover_rrc_once_and_roundtrip(tmp_path):
    documents = request_documents('allreduce', 2)
    documents['topology']['local_costs'] = [
        dict(rank=rank, copy=dict(alpha_us=.2, beta_us_per_byte=.001),
             reduce=dict(alpha_us=.3, beta_us_per_byte=.002)) for rank in (0, 1)]
    inputs = load_request(write_request(tmp_path, documents))
    model = initial_cost_model(inputs)
    link = next(edge for edge in inputs.request_config['topology']['directed_links'] if edge['src'] == 0 and edge['dst'] == 1)
    assert model.services['copy:0'][0].alpha_us == .2
    rrc = model.services['rrc:0:1'][0]
    assert rrc.alpha_us == pytest.approx(link['alpha_us'] + .3)
    assert rrc.beta_us_per_byte == pytest.approx(link['beta_us_per_byte'] + .002)
    assert rrc.source == 'estimated'
    assert model_from_json(model_json(model)) == model


def test_missing_local_costs_stay_missing_and_are_not_zero_filled(tmp_path):
    inputs = load_request(write_request(tmp_path, request_documents('allreduce', 2)))
    model = initial_cost_model(inputs)
    assert 'link:0:1' in model.services
    assert 'copy:0' not in model.services and 'reduce:1' not in model.services and 'rrc:0:1' not in model.services
    assert 'copy:0' in model.metadata['missing_initial_components']


@pytest.mark.parametrize('values', [
    [dict(rank=2, copy=dict(alpha_us=0, beta_us_per_byte=.1))],
    [dict(rank=0, copy=dict(alpha_us=0, beta_us_per_byte=.1))] * 2,
    [dict(rank=0)],
])
def test_invalid_local_cost_identity_rejected_at_input(tmp_path, values):
    documents = request_documents('broadcast', 2)
    documents['topology']['local_costs'] = values
    with pytest.raises(InputValidationError):
        load_request(write_request(tmp_path, documents))


def test_serialized_model_rejects_unknown_fields_and_invalid_components(tmp_path):
    inputs = load_request(write_request(tmp_path, request_documents('broadcast', 2)))
    document = model_json(initial_cost_model(inputs))
    with pytest.raises(InputValidationError):
        model_from_json(document | {'unknown': 1})
    document['services']['link:0:1'][0]['beta_us_per_byte'] = 0
    with pytest.raises(InputValidationError):
        model_from_json(document)
