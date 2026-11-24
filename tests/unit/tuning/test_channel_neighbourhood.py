"""Repair channel enumeration stays linear in the number of changed operations."""
from vericcl.tuning.strategies import _channel_neighbourhood


def test_neighbourhood_starts_with_current_channels_and_changes_one_at_a_time():
    domains = [(1, 0, 2), (0, 1), (3,)]
    assignments = list(_channel_neighbourhood(domains))
    assert assignments[0] == (1, 0, 3)
    assert assignments[1:] == [(0, 0, 3), (2, 0, 3), (1, 1, 3)]
    assert len(list(_channel_neighbourhood([tuple(range(8))] * 10))) == 1 + 10 * 7
