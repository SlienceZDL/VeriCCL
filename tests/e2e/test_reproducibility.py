import pytest

from tests.e2e._support import assert_accepted, documents, sha256, solve


pytestmark = pytest.mark.phase07


def _prediction(run):
    # The record carries a summary; the full timeline is stored once beside
    # the candidate. used_components is a set serialised in hash-seed
    # dependent order.
    import json
    summary = run.record['prediction']
    prediction = json.loads((run.root / summary['detail']).read_text())
    assert {key: prediction[key] for key in summary if key != 'detail'} == {k: v for k, v in summary.items() if k != 'detail'}
    prediction['used_components'] = sorted(prediction['used_components'])
    return prediction


@pytest.mark.parametrize(
    'request_options',
    (
        {'operator': 'allreduce'},
        {'operator': 'allgather', 'ranks': 4, 'slices': 4, 'method': 'hierarchy',
         'gateway_nodes': True, 'stage_num': 3},
    ),
    ids=('direct-allreduce', 'gateway-allgather'),
)
def test_same_inputs_and_seed_reproduce_identical_xml(solved, tmp_path, request_options):
    options = dict(request_options)
    request = documents(options.pop('operator'), seed=0, **options)
    # The first run may be shared with other e2e modules (different case and
    # project root); the second is always a fresh process in a new directory.
    first = solved(request)
    second = solve(tmp_path, request, case='reproduce', run_id='second')

    for run in (first, second):
        assert_accepted(run)
        assert run.record['effective_config']['sketch']['solver']['solver_seed'] == 0
    assert sha256(first.xml_path) == sha256(second.xml_path)
    assert first.xml_text == second.xml_text
    assert first.accepted['xml_sha256'] == second.accepted['xml_sha256']
    assert first.sidecar == second.sidecar
    assert first.record['input_sha256'] == second.record['input_sha256']
    assert _prediction(first) == _prediction(second)
    assert first.accepted['model_version'] == second.accepted['model_version']
    assert first.accepted['t_pred_us'] == second.accepted['t_pred_us']
