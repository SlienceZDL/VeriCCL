from dataclasses import replace
import random

import pytest

from vericcl.errors import SemanticError
from vericcl.execution.budget import StandaloneLease
from vericcl.xml.v5_buffers import _network_order
from vericcl.xml.v5_model import NetworkOperation


def operation(index, dependencies=(), references=(), ranks=(0, 1), priority=0, stage=0):
    return NetworkOperation('n%03d' % index, 'send', *ranks, 0, stage, frozenset({0}),
        frozenset({0}), (0, 1, 2, 3), frozenset(dependencies), frozenset(references), priority)


def scan_reference(operations):
    by_id = {op.operation_id: op for op in operations}
    pending, done, ordered = list(operations), set(), []
    while pending:
        ready = []
        for op in pending:
            references = {key for key in op.reference_dependencies
                          if {op.src_rank, op.dst_rank} & {by_id[key].src_rank, by_id[key].dst_rank}}
            if op.dependencies | references <= done:
                ready.append(op)
        if not ready:
            raise SemanticError('reference cycle')
        selected = min(ready, key=lambda op: (op.priority_time, op.stage_id, op.operation_id))
        ordered.append(selected)
        pending.remove(selected)
        done.add(selected.operation_id)
    return tuple(ordered)


def test_dependency_queue_preserves_exact_scan_order_on_small_dags():
    rng = random.Random(534918)
    for _ in range(60):
        operations = []
        for index in range(32):
            prior = ['n%03d' % earlier for earlier in range(index)]
            dependencies = rng.sample(prior, min(len(prior), rng.randrange(4)))
            references = rng.sample(prior, min(len(prior), rng.randrange(4)))
            ranks = rng.choice([(0, 1), (2, 3), (1, 2)])
            operations.append(operation(index, dependencies, references, ranks, rng.randrange(5), rng.randrange(3)))
        rng.shuffle(operations)
        assert _network_order(operations, StandaloneLease.after(5)) == scan_reference(operations)


def test_ready_priority_is_evaluated_once_per_operation(monkeypatch):
    operations = [operation(index, priority=64 - index) for index in range(64)]
    original = NetworkOperation.__getattribute__
    calls = 0

    def counted(self, key):
        nonlocal calls
        if key == 'priority_time':
            calls += 1
        return original(self, key)

    monkeypatch.setattr(NetworkOperation, '__getattribute__', counted)
    ordered = _network_order(operations, StandaloneLease.after(5))
    assert [op.operation_id for op in ordered] == [op.operation_id for op in reversed(operations)]
    assert calls <= 2 * len(operations), 'unchanged ready operations must not be rescanned every step'


@pytest.mark.parametrize('defect', ['unknown_dependency', 'unknown_reference', 'cycle', 'duplicate'])
def test_invalid_dependency_graphs_remain_rejected(defect):
    operations = [operation(0), operation(1, ['n000'])]
    if defect == 'unknown_dependency':
        operations[0] = replace(operations[0], dependencies=frozenset({'missing'}))
    elif defect == 'unknown_reference':
        operations[0] = replace(operations[0], reference_dependencies=frozenset({'missing'}))
    elif defect == 'cycle':
        operations[0] = replace(operations[0], dependencies=frozenset({'n001'}))
    else:
        operations.append(operations[0])
    with pytest.raises(SemanticError):
        _network_order(operations, StandaloneLease.after(5))
