import json

import pytest

from tests.e2e._support import (
    REQUIRED_CHECKS,
    SLICE_BYTES,
    assert_end_to_end,
    documents,
    recheck,
    sha256,
    solve,
)


pytestmark = pytest.mark.phase07


def test_runtime_limit_rejects_offline_valid_xml_only_at_launch(solved):
    from vericcl.verification.symbolic_check import check_bundle
    from vericcl.xml.capacity import RuntimeLimits
    run = solved(documents('broadcast'))
    longest = max(len(tb.xpath('./step')) for tb in run.xml.xpath('./gpu/tb'))
    assert longest > 1

    bundle, inputs, model = recheck(run, limits=RuntimeLimits(steps_per_tb=longest - 1))
    statuses = {name: check.status.value for name, check in bundle.checks.items()}

    assert not bundle.can_execute
    assert statuses['launch'] == 'invalid'
    assert all(statuses[name] == 'valid' for name in REQUIRED_CHECKS if name != 'launch'), statuses
    assert 'steps_per_tb' in {violation.code for violation in bundle.capacity_report.violations}
    symbolic, _ = check_bundle(bundle, inputs, model)
    assert symbolic.valid, symbolic.message


def test_solve_chooses_a_partition_that_fits_runtime_limits(tmp_path):
    request = documents('broadcast', slices=8)
    request['runtime']['limits'] = {'steps_per_tb': 2}
    run = solve(tmp_path, request)
    assert_end_to_end(run)

    assert max(len(tb.xpath('./step')) for tb in run.xml.xpath('./gpu/tb')) <= 2
    # S is re-chosen for the artifact; the request identity stays unchanged.
    assert run.hyperparameters['total_size_bytes'] == 8 * SLICE_BYTES
    assert run.hyperparameters['slice_size_bytes'] > SLICE_BYTES
    assert run.sidecar['request_contract']['slice_bytes'] == run.hyperparameters['slice_size_bytes']
    effective = json.loads((run.root / 'effective-request.json').read_text())
    assert effective['sketch']['hyperparameters']['slice_size_bytes'] == SLICE_BYTES


def test_every_checked_candidate_is_persisted_with_its_checks(solved):
    run = solved(documents('allreduce'))
    records = [json.loads(path.read_text()) for path in sorted((run.root / 'candidates').glob('*/candidate.json'))]

    assert run.record in records
    for record in records:
        assert set(REQUIRED_CHECKS) <= set(record['checks'])
        if record['can_execute']:
            assert all(record['checks'][name]['status'] == 'valid' for name in REQUIRED_CHECKS)
        if 'xml' in record:
            assert sha256(run.root / record['xml']) == record['xml_sha256']
            assert json.loads((run.root / record['sidecar']).read_text())['xml_sha256'] == record['xml_sha256']
