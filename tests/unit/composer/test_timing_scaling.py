"""The final timing pass shares the same bounded work contract as scheduling."""
import builtins

import pytest

from vericcl.composer import timing
from vericcl.execution.budget import BudgetExhausted, StandaloneLease
from vericcl.solver.global_scheduler import assign_global_resources
from tests.unit.solver.test_global_scheduler_scaling import independent_slices
from tests.unit.verification.simulator_helpers import curve, simulation_topology


@pytest.mark.parametrize('preserve', (False, True))
def test_retime_does_not_recopy_all_completed_ids_for_each_pending_slice(monkeypatch, preserve):
    count = 64
    topology = simulation_topology(2, {(0, 1): curve()}, max_channels=1)
    schedule = assign_global_resources(independent_slices(count), topology, 1)
    copied_entries = 0

    def tracked_set(values=()):
        nonlocal copied_entries
        if isinstance(values, dict):
            copied_entries += len(values)
        return builtins.set(values)

    monkeypatch.setattr(timing, 'set', tracked_set, raising=False)
    result = timing._retime(schedule, topology, preserve_predecessor_order=preserve)
    assert copied_entries <= count * count
    assert result.transfers == schedule.transfers


def test_retime_checks_parent_lease_while_preserving_resource_order():
    topology = simulation_topology(2, {(0, 1): curve()}, max_channels=1)
    schedule = assign_global_resources(independent_slices(64), topology, 1)
    ticks = iter(range(10000))
    lease = StandaloneLease(20., clock=lambda: float(next(ticks)))
    with pytest.raises(BudgetExhausted):
        timing._retime(schedule, topology, preserve_predecessor_order=True, lease=lease)
