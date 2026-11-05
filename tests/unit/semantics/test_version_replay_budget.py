"""A deadline is an interrupted check, not a semantic counterexample."""
import pytest

from vericcl.execution.budget import BudgetExhausted
from vericcl.semantics.version_schedule import replay_version_schedule
from vericcl.semantics.versions import VersionLedger
from vericcl.verification.semantics import check_semantics
from tests.unit.semantics.test_version_schedule import allreduce_schedule


@pytest.mark.parametrize('checker', [replay_version_schedule, check_semantics])
def test_interruption_inside_an_operation_keeps_the_budget_exception(tmp_path, monkeypatch, checker):
    inputs, schedule = allreduce_schedule(tmp_path)

    def interrupted(*args, **kwargs):
        raise BudgetExhausted('semantic replay deadline')

    monkeypatch.setattr(VersionLedger, 'reduce', interrupted)
    with pytest.raises(BudgetExhausted, match='semantic replay deadline'):
        checker(schedule, inputs)


@pytest.mark.parametrize('checker', [replay_version_schedule, check_semantics])
def test_replay_checks_parent_lease_during_initial_state_loading(tmp_path, checker):
    inputs, schedule = allreduce_schedule(tmp_path)

    class Lease:
        calls = 0

        def check(self):
            self.calls += 1
            if self.calls == 2:
                raise BudgetExhausted('initial state deadline')

    lease = Lease()
    with pytest.raises(BudgetExhausted, match='initial state deadline'):
        checker(schedule, inputs, lease=lease)
    assert lease.calls == 2
