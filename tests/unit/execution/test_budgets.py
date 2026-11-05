import pytest

from vericcl.errors import SemanticError


class Clock:
    now = 100.0
    def __call__(self):
        return self.now


def manager(clock, **overrides):
    from vericcl.execution.budget import BudgetManager
    config = dict(stage_limits={'initial': (10, 10), 'repair': (6, 12)},
                  cpu_budget=4, available_cpus=4, max_parallel_models=2,
                  max_threads_per_model=3, per_model_timeout_s=4,
                  cleanup_reserve_s=1, max_tuning_rounds=3, clock=clock)
    config.update(overrides)
    return BudgetManager(**config)


def test_build_and_solve_share_a_model_deadline_and_cleanup_reserve():
    clock = Clock()
    budgets = manager(clock)
    phase = budgets.acquire('initial')
    job = budgets.acquire('initial', model=True, parent=phase, threads=2)
    assert job.deadline == 104
    assert job.work_deadline == 103
    clock.now = 102.5
    assert job.remaining_seconds() == .5
    clock.now = 103
    assert job.expired()
    job.close(cleanup_confirmed=True)
    phase.close(cleanup_confirmed=True)
    assert budgets.cpu_in_use == 0
    assert budgets.consumed['initial'] == 3


def test_queued_models_do_not_reset_parent_time_or_start_their_timer_early():
    from vericcl.execution.budget import ResourceBusy
    clock = Clock()
    budgets = manager(clock, cpu_budget=2, available_cpus=2)
    phase = budgets.acquire('initial')
    first = budgets.acquire('initial', model=True, parent=phase, threads=2)
    with pytest.raises(ResourceBusy):
        budgets.acquire('initial', model=True, parent=phase, threads=1, wait=False)
    clock.now = 102
    first.close(cleanup_confirmed=True)
    second = budgets.acquire('initial', model=True, parent=phase, threads=1)
    assert second.started_at == 102
    assert second.deadline == 106
    assert phase.deadline == 110


def test_unconfirmed_exit_keeps_cpu_and_blocks_parent_completion():
    from vericcl.execution.budget import ResourceBusy
    clock = Clock()
    budgets = manager(clock)
    phase = budgets.acquire('initial')
    job = budgets.acquire('initial', model=True, parent=phase, threads=3)
    job.close(cleanup_confirmed=False)
    assert budgets.cpu_in_use == 3
    with pytest.raises(ResourceBusy):
        phase.close(cleanup_confirmed=True)
    job.close(cleanup_confirmed=True)
    phase.close(cleanup_confirmed=True)
    assert budgets.cpu_in_use == 0


def test_cumulative_wall_time_and_rounds_survive_multiple_attempts():
    from vericcl.execution.budget import BudgetExhausted
    clock = Clock()
    budgets = manager(clock)
    for _ in range(2):
        budgets.start_tuning_round()
        phase = budgets.acquire('repair')
        clock.now += 6
        phase.close(cleanup_confirmed=True)
    assert budgets.started_rounds == 2
    with pytest.raises(BudgetExhausted):
        budgets.start_tuning_round()
    assert budgets.snapshot()['consumed']['repair'] == 12


def test_global_wall_limit_and_model_cancellation_bound_new_work():
    from vericcl.execution.budget import BudgetExhausted
    clock = Clock()
    budgets = manager(clock, max_run_wall_s=5)
    phase = budgets.acquire('initial')
    assert phase.work_deadline == 104
    clock.now = 104
    with pytest.raises(BudgetExhausted):
        budgets.acquire('initial', model=True, parent=phase)
    budgets.cancel()
    assert phase.cancelled


def test_restore_never_refunds_consumed_time_or_started_rounds():
    clock = Clock()
    budgets = manager(clock)
    budgets.restore_usage({'initial': 4, 'repair': 3}, started_rounds=2, elapsed_wall_s=20)
    assert budgets.acquire('initial').deadline == 106
    with pytest.raises(SemanticError):
        budgets.restore_usage({'initial': 1}, started_rounds=1, elapsed_wall_s=1)


def test_multiple_actions_share_one_outer_round_phase_allowance():
    from vericcl.execution.budget import BudgetExhausted
    clock = Clock()
    budgets = manager(clock)
    budgets.start_tuning_round()
    first = budgets.acquire('repair')
    clock.now += 3
    first.close(cleanup_confirmed=True)
    second = budgets.acquire('repair')
    assert second.deadline == 106
    clock.now += 2.2
    second.close(cleanup_confirmed=True)
    with pytest.raises(BudgetExhausted):
        budgets.acquire('repair')
    budgets.start_tuning_round()
    assert budgets.acquire('repair').deadline == pytest.approx(111.2)
