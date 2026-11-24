from vericcl.performance.model import CostModel, ServiceCost
from vericcl.performance.profiles import RateProfile
from vericcl.live.triggers import TriggerEvent


def test_live_drift_changes_only_observed_size_and_preserves_unobserved_links():
    from vericcl.live.model import update_live_model
    model = CostModel('before', {'link:0:1': (ServiceCost(2, .01),), 'link:1:0': (ServiceCost(3, .02),)},
        {'link:0:1': 100, 'link:1:0': 50})
    event = TriggerEvent('performance', 24, 'xml', {'service:link:0:1': .2, 'request': .3})
    after = update_live_model(model, (event,), payload_bytes=1024)
    assert after.version != model.version
    assert after.service_for('link:0:1', 1024).alpha_us == 2.4
    assert after.service_for('link:0:1', 1024).source == 'estimated'
    assert after.service_for('link:0:1', 512).alpha_us == 2
    assert after.service_for('link:1:0', 1024) == model.service_for('link:1:0', 1024)
    assert after.metadata['live_update']['scope'] == 'observed_payload_only_relative_drift_estimate'


def test_existing_concurrency_profile_is_updated_without_inventing_new_modes():
    from vericcl.live.model import update_live_model
    profile = RateProfile('p', ('send:0:1:c0',), {'send:0:1:c0': 100}, ('link:0:1',),
        source='initial', flow_resources={'send:0:1:c0': ('link:0:1',)})
    model = CostModel('before', {'link:0:1': (ServiceCost(2, .01),)}, {'link:0:1': 100}, (profile,))
    after = update_live_model(model, (TriggerEvent('performance', 24, 'xml', {'service:link:0:1': 1}),), payload_bytes=1024)
    point = next(p for p in after.profiles if p.min_bytes == p.max_bytes == 1024)
    assert point.rates['send:0:1:c0'] == 50 and point.source == 'estimated'
    assert {p.mode for p in after.profiles} == {profile.mode}
