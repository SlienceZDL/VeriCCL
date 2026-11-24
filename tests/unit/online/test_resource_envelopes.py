"""Shared-resource capacities from measured profiles: lower bounds unless saturated."""
from fractions import Fraction

from vericcl.performance.profiles import RateProfile
from vericcl.verification.online.model_publication import resource_envelopes


def profile(rates, resource='r'):
    flows = tuple(rates)
    return RateProfile('p-' + '-'.join(flows), flows, dict(rates), (resource,), 1, 100, 'measured',
                       '2026-11-23T00:00:00+00:00', {flow: 1.0 for flow in flows},
                       {flow: (resource,) for flow in flows})


def test_single_flow_rate_never_lowers_a_shared_capacity():
    assert resource_envelopes([profile({'a': 5.0})], {'r': 14.0}) == {'r': Fraction(14)}


def test_growing_aggregate_is_only_a_lower_bound():
    profiles = [profile({'a': 5.0}), profile({'a': 4.5, 'b': 4.5})]
    assert resource_envelopes(profiles, {'r': 14.0}) == {'r': Fraction(14)}
    assert resource_envelopes(profiles, {'r': 8.0}) == {'r': Fraction(9)}


def test_saturated_aggregate_defines_the_capacity():
    profiles = [profile({'a': 4.5, 'b': 4.5}), profile({'a': 2.4, 'b': 2.4, 'c': 2.4, 'd': 2.4})]
    assert resource_envelopes(profiles, {'r': 14.0}) == {'r': Fraction('9.6')}
