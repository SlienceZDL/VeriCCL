from dataclasses import replace
from datetime import datetime, timezone

import pytest

from vericcl.performance.model import CostModel, ServiceCost


def test_partial_publication_preserves_applicable_components_and_original_input():
    from vericcl.verification.online.model_publication import ComponentUpdates, publish_model
    initial = CostModel('initial', {'link:0:1': (ServiceCost(1, .1),), 'copy:0': (ServiceCost(1, .2),)})
    measured = ServiceCost(2, .03, 'measured', 100, 1000, '2026-11-04T00:00:00Z')
    updates = ComponentUpdates(services={'link:0:1': (measured,)}, signatures={'service:link:0:1': 'edge-A'})
    updated = publish_model(initial, updates, initial=initial)
    assert updated.version != initial.version
    assert updated.service_for('link:0:1', 500) == measured
    assert updated.service_for('link:0:1', 10).source == 'initial'
    assert updated.services['copy:0'] == initial.services['copy:0']
    assert initial.service_for('link:0:1', 500).beta_us_per_byte == .1
    again = publish_model(updated, ComponentUpdates(issues=({'component': 'shared', 'reason': 'failed'},)), initial=initial)
    assert again is updated


def test_changed_applicability_invalidates_only_that_component_and_uses_initial():
    from vericcl.verification.online.model_publication import ComponentUpdates, publish_model
    stamp = '2026-11-04T00:00:00Z'
    initial = CostModel('initial', {'link:0:1': (ServiceCost(1, .1),), 'copy:0': (ServiceCost(1, .2),)})
    updates = ComponentUpdates(services={'link:0:1': (ServiceCost(2, .03, 'measured', 1, None, stamp),),
                                        'copy:0': (ServiceCost(2, .04, 'measured', 1, None, stamp),)},
                               signatures={'service:link:0:1': 'edge-A', 'service:copy:0': 'gpu-A'})
    old = publish_model(initial, updates, initial=initial)
    new = publish_model(old, ComponentUpdates(), initial=initial,
                        current_signatures={'service:link:0:1': 'edge-B', 'service:copy:0': 'gpu-A'})
    assert new.service_for('link:0:1', 10).source == 'initial'
    assert new.service_for('copy:0', 10).source == 'measured'


def test_cross_task_cache_is_opt_in_and_does_not_refresh_original_measurement_age(tmp_path):
    from vericcl.verification.online.component_cache import CalibrationQuery, lookup_calibration, save_calibration
    value = ServiceCost(2, .03, 'measured', 100, 1000, '2026-11-04T00:00:00Z')
    path = tmp_path / 'cost-cache.json'
    save_calibration(path, 'service:link:0:1', 'identity-A', (value,))
    now = datetime(2026, 11, 4, 0, 10, tzinfo=timezone.utc)
    query = CalibrationQuery(path, 'service:link:0:1', 'identity-A', now=now, enabled=False, max_age_s=1000)
    assert lookup_calibration(query).status == 'disabled'
    hit = lookup_calibration(replace(query, enabled=True))
    assert hit.status == 'hit' and hit.values[0].source == 'history'
    assert hit.values[0].measured_at == value.measured_at
    save_calibration(path, 'service:link:0:1', 'identity-A', hit.values)
    assert lookup_calibration(replace(query, enabled=True)).values[0].measured_at == value.measured_at
    assert lookup_calibration(replace(query, enabled=True, max_age_s=100)).status == 'miss'
    assert lookup_calibration(replace(query, enabled=True, signature='different-datatype')).status == 'miss'
    path.write_text('{broken')
    assert lookup_calibration(replace(query, enabled=True)).status == 'invalid_cache'


def test_absolute_calibration_cache_does_not_accept_ratio_only_matching(tmp_path):
    from vericcl.verification.online.component_cache import CalibrationQuery
    with pytest.raises(ValueError):
        CalibrationQuery(tmp_path / 'cache', 'service:x', 'identity', enabled=True, max_age_s=None)


def test_stale_new_measurement_cannot_override_a_current_identity():
    from vericcl.verification.online.model_publication import ComponentUpdates, publish_model
    initial = CostModel('initial', {'copy:0': (ServiceCost(1, .1),)})
    updates = ComponentUpdates(services={'copy:0': (ServiceCost(2, .03, 'measured', 1, None, '2026-11-04T00:00:00Z'),)},
                               signatures={'service:copy:0': 'old-runtime'})
    result = publish_model(initial, updates, initial=initial, current_signatures={'service:copy:0': 'current-runtime'})
    assert result.service_for('copy:0', 100).source == 'initial'
