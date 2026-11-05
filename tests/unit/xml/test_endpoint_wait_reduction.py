from types import SimpleNamespace

from tests.integration.test_native_endpoint_order import local_predecessors
from tests.unit.input.v5_support import request_documents, write_request
from vericcl.execution.budget import StandaloneLease
from vericcl.input.loader import load_request
from vericcl.xml.v5_model import Address, Instruction
from vericcl.xml.v5_program import lower_program


def test_repeated_local_waits_preserve_all_dependencies_without_quadratic_nops(tmp_path):
    count = 64
    documents = request_documents('broadcast', ranks=3)
    documents['sketch']['hyperparameters'].update(total_size_bytes=count * 1024, slice_size_bytes=1024)
    inputs = load_request(write_request(tmp_path, documents))
    instructions, addresses, previous = [], {}, []
    for index in range(count):
        source, target = 'input-%d' % index, 'root-%d' % index
        addresses[source], addresses[target] = Address(0, 'i', index), Address(0, 'o', index)
        identifier = 'copy-%d' % index
        instructions.append(Instruction(identifier, 'copy', (source,), target, 0, -1, 0, 0, (0,), frozenset(), 0))
        previous.append(identifier)
    for peer in (1, 2):
        dependencies = frozenset(previous)
        for index in range(count):
            source, target = 'root-%d' % index, 'peer-%d-%d' % (peer, index)
            addresses[target] = Address(peer, 'o', index)
            identifier = 'send-%d-%d' % (peer, index)
            instructions.append(Instruction(identifier, 'send', (source,), target, 0, peer, 0, 0,
                                            (0, peer), dependencies, 0))
            previous.append(identifier)
    storage = SimpleNamespace(instructions=tuple(instructions), addresses=addresses,
                              input_slots={0: count, 1: 0, 2: 0},
                              output_slots={rank: count for rank in range(3)}, scratch_slots={})
    context = SimpleNamespace(inputs=inputs, lease=StandaloneLease.after(30))
    program = lower_program(storage, context)
    # Every dependency is already implied by one latest local endpoint or the
    # preceding work on this TB. These 64-slice broadcasts need no NOP chain.
    assert max(len(block.steps) for block in program.blocks) == count
    assert not any(step.kind == 'nop' for block in program.blocks for step in block.steps)
    locations = {(step.operation_id, block.rank): (block.rank, block.tb_id, index)
                 for block in program.blocks for index, step in enumerate(block.steps)}
    for instruction in instructions:
        for rank in (0, instruction.peer) if instruction.kind == 'send' else (0,):
            location = locations[instruction.operation_id, rank]
            ancestors = local_predecessors(program, *location)
            for predecessor in instruction.dependencies:
                if (predecessor, rank) in locations:
                    assert locations[predecessor, rank] in ancestors
