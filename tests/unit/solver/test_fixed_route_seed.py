"""Check the compact seed against an independent exhaustive greedy oracle."""
import random
from types import SimpleNamespace

import pytest

from vericcl.errors import SemanticError
from vericcl.execution.budget import BudgetExhausted
from vericcl.solver.trees import timing


def operation(index, *, edge=(0, 1), resources=(), dependencies=(), kind='send', demand=None):
    return SimpleNamespace(operation_id='op-%04d' % index, edge=edge, resources=tuple(resources),
        dependencies=frozenset(dependencies), kind=kind, demand_id=demand or str(index))


def oracle(operations, domains, durations):
    pending = {op.operation_id: op for op in operations}
    selected, dependencies, starts, finishes, completed = {}, {}, {}, {}, []
    while pending:
        choices = []
        for identifier, op in pending.items():
            if not op.dependencies <= finishes.keys():
                continue
            for channel in domains[identifier]:
                predecessors = set(op.dependencies)
                for previous in completed:
                    same_accumulator = (op.kind == previous.kind == 'rrc'
                        and op.demand_id == previous.demand_id and op.edge[1] == previous.edge[1])
                    shared = op.edge == previous.edge or bool(set(op.resources) & set(previous.resources))
                    if same_accumulator or (shared and selected[previous.operation_id] == channel):
                        predecessors.add(previous.operation_id)
                start = max((finishes[p] for p in predecessors), default=0.)
                choices.append((start + durations[identifier], start, identifier, channel, predecessors))
        end, start, identifier, channel, predecessors = min(choices, key=lambda item: item[:4])
        selected[identifier], dependencies[identifier] = channel, predecessors
        starts[identifier], finishes[identifier] = start, end
        completed.append(pending.pop(identifier))
    return SimpleNamespace(channels=selected, dependencies=dependencies, start_us=starts, finish_us=finishes)


def ancestors(dependencies, identifier):
    reached, pending = set(), list(dependencies[identifier])
    while pending:
        item = pending.pop()
        if item not in reached:
            reached.add(item)
            pending.extend(dependencies[item])
    return reached


@pytest.mark.parametrize('seed', range(30))
def test_seed_preserves_greedy_channels_times_and_dependency_reachability(seed):
    rng = random.Random(seed)
    operations = []
    for index in range(25):
        predecessors = [op.operation_id for op in operations if rng.random() < .12]
        operations.append(operation(index, edge=(rng.randrange(3), rng.randrange(3, 6)),
            resources=[r for r in ('nic', 'nvlink') if rng.random() < .4], dependencies=predecessors,
            kind=rng.choice(('send', 'rrc')), demand=str(rng.randrange(3))))
    rng.shuffle(operations)
    domains = {op.operation_id: tuple(range(rng.randrange(1, 4))) for op in operations}
    durations = {op.operation_id: rng.randrange(1, 9) / 4 for op in operations}
    expected = oracle(operations, domains, durations)
    actual = timing._initial(SimpleNamespace(operations=operations), domains, durations)
    assert dict(actual.channels) == expected.channels
    assert dict(actual.start_us) == expected.start_us
    assert dict(actual.finish_us) == expected.finish_us
    for op in operations:
        assert ancestors(actual.dependencies, op.operation_id) == ancestors(expected.dependencies, op.operation_id)


def test_many_slices_export_linear_resource_precedence_without_losing_order():
    operations = [operation(index, resources=('nic',)) for index in range(96)]
    domains = {op.operation_id: (0,) for op in operations}
    durations = {op.operation_id: 1. for op in operations}
    actual = timing._initial(SimpleNamespace(operations=operations), domains, durations)
    assert max(actual.finish_us.values()) == len(operations)
    assert sum(map(len, actual.dependencies.values())) == len(operations) - 1
    assert ancestors(actual.dependencies, operations[-1].operation_id) == set(domains) - {operations[-1].operation_id}


@pytest.mark.parametrize('dependencies', [{'a': {'b'}, 'b': {'a'}}, {'a': {'missing'}}])
def test_earliest_rejects_cycles_and_missing_predecessors(dependencies):
    operations = [SimpleNamespace(operation_id=identifier) for identifier in dependencies]
    with pytest.raises(SemanticError, match='dependenc'):
        timing._earliest(operations, dict.fromkeys(dependencies, 0), dependencies, dict.fromkeys(dependencies, 1.))


def test_earliest_replay_obeys_parent_cancellation():
    class Cancelled:
        def check(self):
            raise BudgetExhausted('cancelled during timing replay')
    op = operation(0)
    with pytest.raises(BudgetExhausted, match='cancelled during timing replay'):
        timing._earliest([op], {op.operation_id: 0}, {op.operation_id: set()}, {op.operation_id: 1.}, Cancelled())
