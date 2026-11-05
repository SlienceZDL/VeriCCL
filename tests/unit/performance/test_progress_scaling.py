"""Readiness follows dependency edges instead of rescanning every pending operation."""
import builtins

import pytest

from vericcl.performance import simulation
from vericcl.performance.events import Executable, WorkOperation
from tests.unit.performance.test_event_progress import model


@pytest.mark.parametrize('exact', (False, True))
def test_dependency_chain_has_linear_dependency_bookkeeping(monkeypatch, exact):
    count = 128
    operations = tuple(WorkOperation('op-%03d' % i, 'copy', 'wire', 10,
        dependencies=() if i == 0 else ('op-%03d' % (i-1),)) for i in range(count))
    executable = Executable(operations)
    dependency_entries = 0

    def observed_set(values=()):
        nonlocal dependency_entries
        if isinstance(values, tuple):
            dependency_entries += len(values)
        return builtins.set(values)

    monkeypatch.setattr(simulation, 'set', observed_set, raising=False)
    result = simulation.predict(executable, model(alpha=1, beta=.1), exact=exact)
    assert result.t_pred_us == 256
    assert result.start_times['op-127'] == 254
    assert dependency_entries <= 10 * count


@pytest.mark.parametrize('exact', (False, True))
def test_timed_release_and_nop_chain_preserve_event_order(exact):
    executable = Executable((
        WorkOperation('a', 'copy', 'wire', 10),
        WorkOperation('b', 'nop', None, 0, dependencies=('a',)),
        WorkOperation('c', 'nop', None, 0, dependencies=('b',)),
        WorkOperation('d', 'copy', 'wire', 10, dependencies=('c',), not_before_us=5),
        WorkOperation('e', 'copy', 'wire', 10, not_before_us=5),
        WorkOperation('f', 'copy', 'wire', 10, dependencies=('d', 'e')),
    ))
    result = simulation.predict(executable, model(), exact=exact)
    assert result.start_times == {'a': 0., 'b': 1., 'c': 1., 'd': 5., 'e': 5., 'f': 6.}
    assert result.end_times == {'a': 1., 'b': 1., 'c': 1., 'd': 6., 'e': 6., 'f': 7.}
    assert [(event.time_us, event.kind, event.operation_id) for event in result.events] == [
        (0., 'start', 'a'), (0., 'startup_complete', 'a'), (1., 'complete', 'a'),
        (1., 'start', 'b'), (1., 'complete', 'b'), (1., 'start', 'c'), (1., 'complete', 'c'),
        (5., 'start', 'd'), (5., 'start', 'e'), (5., 'startup_complete', 'd'),
        (5., 'startup_complete', 'e'), (6., 'complete', 'd'), (6., 'complete', 'e'),
        (6., 'start', 'f'), (6., 'startup_complete', 'f'), (7., 'complete', 'f')]
