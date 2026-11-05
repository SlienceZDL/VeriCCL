"""A receive wait cannot protect an earlier source read on the sending GPU."""
from dataclasses import replace

import pytest

from tests.integration.test_lowering_v5 import candidate_fixture
from vericcl.errors import SemanticError
from vericcl.xml.lower import lower_and_check


def local_predecessors(program, rank, tb, index):
    steps = {(b.rank, b.tb_id, i): s for b in program.blocks for i, s in enumerate(b.steps)}
    pending, visited = [(rank, tb, index)], set()
    while pending:
        key = pending.pop()
        if key in visited:
            continue
        visited.add(key)
        step = steps[key]
        if key[2]:
            pending.append((key[0], key[1], key[2] - 1))
        if step.dependency_tb >= 0:
            pending.append((key[0], step.dependency_tb, step.dependency_step))
    return visited - {(rank, tb, index)}


def test_allreduce_sender_waits_for_local_reduction_result(tmp_path):
    candidate, context = candidate_fixture(tmp_path, 'allreduce')
    bundle = lower_and_check(candidate, context)
    assert bundle.can_execute
    producers = {(b.rank, s.dst): (b.rank, b.tb_id, i)
                 for b in bundle.program.blocks for i, s in enumerate(b.steps) if s.kind == 'rrc'}
    checked = 0
    for block in bundle.program.blocks:
        for index, step in enumerate(block.steps):
            if step.kind == 's' and (block.rank, step.src) in producers:
                assert producers[block.rank, step.src] in local_predecessors(bundle.program, block.rank, block.tb_id, index)
                checked += 1
    assert checked


def test_receiver_wait_alone_does_not_validate_the_sender_data_dependency(tmp_path):
    from vericcl.xml.v5_lower import _validate_bindings
    candidate, context = candidate_fixture(tmp_path, 'allreduce')
    bundle = lower_and_check(candidate, context)
    # Keep the receiver's dependency and remove all waits from the root sender.
    broken = replace(bundle.program, blocks=tuple(replace(block, steps=tuple(
        replace(step, dependency_tb=-1, dependency_step=-1) if block.rank == 0 and step.kind == 's' else step
        for step in block.steps)) for block in bundle.program.blocks))
    with pytest.raises(SemanticError, match='local endpoint'):
        _validate_bindings(broken, bundle.storage_plan, context)
