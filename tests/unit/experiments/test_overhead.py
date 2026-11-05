import sys
from types import ModuleType

import pytest


def test_nested_timing_partitions_wall_time_without_double_counting():
    from vericcl.experiments.overhead import PhaseTimer

    ticks = iter((0, 10, 30, 50))
    timer = PhaseTimer(clock=lambda: next(ticks))
    with timer.span("workflow", priority=0):
        with timer.span("validation", priority=10):
            pass
    assert timer.breakdown_ns() == {"workflow": 30, "validation": 20}


def test_timing_records_exceptions_without_suppressing_them():
    from vericcl.experiments.overhead import PhaseTimer

    ticks = iter((2, 9))
    timer = PhaseTimer(clock=lambda: next(ticks))
    with pytest.raises(ValueError, match="failure"):
        with timer.span("repair"):
            raise ValueError("failure")
    assert timer.breakdown_ns() == {"repair": 7}


def test_instrumentation_covers_aliases_and_restores_originals():
    from vericcl.experiments.overhead import PhaseTimer, instrument

    def operation(value):
        return value + 1

    module = ModuleType("vericcl._overhead_test")
    module.first = operation
    module.second = operation
    sys.modules[module.__name__] = module
    timer = PhaseTimer()
    try:
        with instrument(timer, ((operation, "operation", 10),)):
            assert module.first(4) == 5
            assert module.second(8) == 9
        assert module.first is operation
        assert module.second is operation
        assert len(timer.events) == 2
    finally:
        del sys.modules[module.__name__]


def test_repeat_count_rejects_zero_and_warmup_is_excluded():
    from vericcl.experiments.overhead import repeated

    calls = []
    with pytest.raises(ValueError):
        repeated(lambda: None, repeats=0)
    result = repeated(lambda: calls.append(len(calls)), repeats=3)
    assert len(calls) == 4
    assert len(result["samples_s"]) == 3
    assert result["warmups"] == 1
