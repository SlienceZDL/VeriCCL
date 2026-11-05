"""Native graph validation preserves all predecessor paths and parent deadlines."""
import random

import pytest

from vericcl.errors import SemanticError
from vericcl.execution.budget import BudgetExhausted
from vericcl.xml import v5_program
from vericcl.xml.v5_model import Address, NativeBlock, NativeProgram, NativeStep


def reference_ancestors(graph, key):
    pending, seen = list(graph[key]), set()
    while pending:
        predecessor = pending.pop()
        if predecessor not in seen:
            seen.add(predecessor)
            pending.extend(graph[predecessor])
    return seen


def test_native_transitive_closure_matches_independent_path_walk():
    randomizer = random.Random(765134)
    for _ in range(80):
        keys = [(randomizer.randrange(4), index) for index in range(48)]
        graph = {key: set(randomizer.sample(keys[:index], min(index, randomizer.randrange(5))))
                 for index, key in enumerate(keys)}
        order = list(graph)
        randomizer.shuffle(order)
        graph = {key: graph[key] for key in order}
        ancestors, bits = v5_program._dependency_ancestors(graph)
        for key in graph:
            actual = {predecessor for predecessor, bit in bits.items() if ancestors[key] & bit}
            assert actual == reference_ancestors(graph, key)


@pytest.mark.parametrize('graph', [
    {'a': {'a'}}, {'a': {'b'}, 'b': {'a'}}, {'a': set(), 'b': {'missing'}},
])
def test_invalid_dependency_closures_remain_rejected(graph):
    with pytest.raises(SemanticError):
        v5_program._dependency_ancestors(graph)


class ExpiredLease:
    def check(self):
        raise BudgetExhausted('parent deadline')


def test_native_graph_checks_the_parent_lease_before_work():
    steps = tuple(NativeStep('step-' + str(index), 'copy-' + str(index), 'cpy',
        Address(0, 'i', index), Address(0, 'o', index)) for index in range(128))
    program = NativeProgram(1, 1, 128, (NativeBlock(0, 0, 0, -1, -1, steps),),
                            {0: 128}, {0: 128}, {0: 0})
    with pytest.raises(BudgetExhausted, match='parent deadline'):
        v5_program.execution_dependencies(program, lease=ExpiredLease())


def test_dependency_closure_checks_the_parent_lease():
    with pytest.raises(BudgetExhausted, match='parent deadline'):
        v5_program._dependency_ancestors({'a': set()}, lease=ExpiredLease())


def test_native_copy_chain_keeps_each_immediate_predecessor():
    steps = tuple(NativeStep('step-' + str(index), 'copy-' + str(index), 'cpy',
        Address(0, 'i', index), Address(0, 'o', index)) for index in range(512))
    program = NativeProgram(1, 1, 512, (NativeBlock(0, 0, 0, -1, -1, steps),),
                            {0: 512}, {0: 512}, {0: 0})
    dependencies, endpoints = v5_program.execution_dependencies(program)
    assert dependencies == {'copy-' + str(index): frozenset(() if index == 0 else ('copy-' + str(index-1),))
                            for index in range(512)}
    assert len(endpoints) == 512
