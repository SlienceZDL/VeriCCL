import hashlib
from dataclasses import replace

import pytest


@pytest.mark.parametrize('operator', ['broadcast', 'reduce', 'allgather', 'allreduce', 'alltoall', 'reduce_scatter'])
def test_manifest_is_bound_to_checked_xml_and_actual_geometry(tmp_path, operator):
    from tests.integration.test_lowering_v5 import candidate_fixture
    from vericcl.xml.lower import lower_and_check
    from vericcl.live.manifest import export_manifest
    candidate, lowering = candidate_fixture(tmp_path, operator)
    bundle = lower_and_check(candidate, lowering)
    manifest = export_manifest(bundle, lowering.inputs, tmp_path / 'schedule.xml')
    assert manifest['xml_sha256'] == hashlib.sha256(bundle.xml_text.encode()).hexdigest()
    assert manifest['collective'] == operator
    assert manifest['runtime_bytes'] == bundle.program.chunks_per_loop * lowering.inputs.hyperparameters.slice_size_bytes
    assert len(manifest['ranks']) == lowering.inputs.rank_count
    assert all(item['input_bytes'] == bundle.program.input_slots[item['rank']] * manifest['slice_bytes'] for item in manifest['ranks'])
    with pytest.raises(ValueError, match='qualified'):
        export_manifest(replace(bundle, checks={}), lowering.inputs, tmp_path / 'bad.xml')


def test_trace_capacity_must_cover_full_invocation_and_never_change_xml(tmp_path):
    from tests.integration.test_lowering_v5 import candidate_fixture
    from vericcl.xml.lower import lower_and_check
    from vericcl.live.manifest import export_manifest
    candidate, lowering = candidate_fixture(tmp_path, 'broadcast')
    bundle = lower_and_check(candidate, lowering)
    base = export_manifest(bundle, lowering.inputs, tmp_path / 'schedule.xml')
    trace = export_manifest(bundle, lowering.inputs, tmp_path / 'schedule.xml', trace_records={0: 256, 1: 256}, trace_prefix=tmp_path / 'trace')
    assert base['version_id'] == trace['version_id']
    assert base['contract_digest'] == trace['contract_digest']
    with pytest.raises(ValueError, match='capacity'):
        export_manifest(bundle, lowering.inputs, tmp_path / 'schedule.xml', trace_records={0: 0, 1: 0}, trace_prefix=tmp_path / 'trace')


def test_inplace_dependency_nops_use_compacted_native_counter_positions(tmp_path):
    from tests.integration.test_lowering_v5 import candidate_fixture
    from vericcl.xml.lower import lower_and_check
    from vericcl.xml.trace_sidecar import build_v5_trace_sidecar
    from vericcl.live.manifest import export_manifest
    from vericcl.xml.v5_lower import evaluate_program
    from vericcl.xml.v5_model import NativeStep
    from vericcl.xml.v5_program import emit_native_xml
    candidate, lowering = candidate_fixture(tmp_path, 'allreduce', inplace=True)
    bundle = lower_and_check(candidate, lowering)
    # The lowering optimizer can remove every redundant wait in this example.
    # Insert a legal explicit NOP to exercise the runtime's compacted counter
    # positions independently of the optimizer's choice of dependency form.
    owner, position, target = next((block, index, step) for block in bundle.program.blocks
        for index, step in enumerate(block.steps) if step.kind != 'nop' and step.dependency_tb >= 0)
    blocks = []
    for block in bundle.program.blocks:
        steps = list(block.steps)
        if block is owner:
            steps.insert(position, NativeStep('counter-nop-fixture', target.operation_id, 'nop', None, None,
                                              target.dependency_tb, target.dependency_step))
        steps = [replace(step, dependency_step=step.dependency_step + 1)
                 if block.rank == owner.rank and step.dependency_tb == owner.tb_id and step.dependency_step >= position
                 else step for step in steps]
        blocks.append(replace(block, steps=tuple(steps)))
    program = replace(bundle.program, blocks=tuple(blocks))
    bundle = evaluate_program(candidate, bundle.storage_plan, program, emit_native_xml(program, lowering.inputs), lowering)
    assert bundle.can_execute
    assert any(step.kind == 'nop' for block in bundle.program.blocks for step in block.steps)
    manifest = export_manifest(bundle, lowering.inputs, tmp_path / 'schedule.xml')
    trace = build_v5_trace_sidecar(bundle)
    actual = {(rank['rank'], step['tb_id'], step['step_index']): step['operation_id']
              for rank in manifest['ranks'] for step in rank['steps']}
    expected = {(entry.rank, entry.tb_id, entry.step_index): entry.transfer_id for entry in trace.entries.values()}
    assert actual == expected
    assert all(step['kind'] != 'nop' for rank in manifest['ranks'] for step in rank['steps'])
