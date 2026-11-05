"""Bound bookkeeping work and honour the synthesis budget after route solving."""
import builtins

import pytest

from vericcl.execution.budget import BudgetExhausted, StandaloneLease
from vericcl.solver import global_scheduler
from tests.unit.solver.test_global_scheduler import _schedule, _transfer
from tests.unit.verification.simulator_helpers import curve, simulation_topology


def independent_slices(count, *, hierarchy=False):
    return _schedule(tuple(_transfer('slice-%04d' % index, slice_id=index,
        slice_count=count, path=((0, 'SEND', ((0, 1, 0.),)),))
        for index in range(count)), rank_count=2, slice_count=count,
        metadata={'hierarchy_event_policy': hierarchy})


@pytest.mark.parametrize('hierarchy', (False, True))
def test_ready_scan_does_not_copy_completed_ids_for_every_pending_slice(monkeypatch, hierarchy):
    count = 64
    schedule = independent_slices(count, hierarchy=hierarchy)
    topology = simulation_topology(2, {(0, 1): curve()}, max_channels=1)
    copied_entries = 0

    def tracked_set(values=()):
        nonlocal copied_entries
        if isinstance(values, dict):
            copied_entries += len(values)
        return builtins.set(values)

    monkeypatch.setattr(global_scheduler, 'set', tracked_set, raising=False)
    result = global_scheduler.assign_global_resources(schedule, topology, 1)

    assert copied_entries <= count * count
    ordered = sorted(result.transfers, key=lambda item: item.st_time)
    assert [item.transfer_id for item in ordered] == sorted(item.transfer_id for item in schedule.transfers)
    assert all(left.ed_time <= right.st_time for left, right in zip(ordered, ordered[1:]))


def test_cpu_resource_assignment_checks_parent_lease_during_scheduling():
    schedule = independent_slices(64, hierarchy=True)
    topology = simulation_topology(2, {(0, 1): curve()}, max_channels=1)
    ticks = iter(range(10000))
    lease = StandaloneLease(20., clock=lambda: float(next(ticks)))
    with pytest.raises(BudgetExhausted):
        global_scheduler.assign_global_resources(schedule, topology, 1, lease=lease)


@pytest.mark.parametrize('hierarchy', (False, True))
def test_equal_resource_slices_do_not_repeat_all_ready_resource_queries(monkeypatch, hierarchy):
    count = 128
    schedule = independent_slices(count, hierarchy=hierarchy)
    topology = simulation_topology(2, {(0, 1): curve()}, max_channels=1)
    original = global_scheduler._earliest_resource_slots
    queries = 0

    def observed(*args, **kwargs):
        nonlocal queries
        queries += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(global_scheduler, '_earliest_resource_slots', observed)
    result = global_scheduler.assign_global_resources(schedule, topology, 1)
    assert len(result.transfers) == count
    assert queries <= 4 * count
    assert [item.transfer_id for item in sorted(result.transfers, key=lambda item: item.st_time)] == [
        'slice-%04d' % index for index in range(count)]

