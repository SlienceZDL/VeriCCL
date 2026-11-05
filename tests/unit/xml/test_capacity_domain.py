from dataclasses import replace

import pytest

from vericcl.errors import InputValidationError, SemanticError
from vericcl.input.loader import load_request
from tests.unit.input.v5_support import request_documents, write_request


def block(step_count=1, **kwargs):
    from vericcl.xml.capacity import BlockCapacity, StepCapacity
    fields = dict(rank=0, tb_id=0, channel=0, send_peer=1, recv_peer=-1,
                  steps=tuple(StepCapacity('s', src_offset=0) for _ in range(step_count)))
    fields.update(kwargs)
    return BlockCapacity(**fields)


def report(blocks, **kwargs):
    from vericcl.xml.capacity import ExecutionShape, RuntimeLimits, check_capacity
    return check_capacity(ExecutionShape(2, 1, tuple(blocks)), kwargs.pop('limits', RuntimeLimits()), **kwargs)


def test_nop_steps_count_toward_the_exact_step_limit():
    from vericcl.xml.capacity import StepCapacity
    assert report([block(256)]).compatible
    result = report([
        block(step_count=0, **{'steps': tuple([StepCapacity('s', src_offset=0)] * 256 + [StepCapacity('nop')])})])
    assert 'steps_per_tb' in {v.code for v in result.violations}


@pytest.mark.parametrize('field,value,code', [
    ('src_offset', 32768, 'buffer_offset'), ('dependency_tb', 128, 'dependency_id'),
    ('dependency_step', 256, 'dependency_step'), ('count', 2, 'runtime_count'),
])
def test_typed_step_limits_are_checked_before_lowering(field, value, code):
    from vericcl.xml.capacity import StepCapacity
    step = replace(StepCapacity('s', src_offset=0), **{field: value})
    result = report([block(step_count=0, **{'steps': (step,)})])
    assert code in {v.code for v in result.violations}


def test_connections_have_one_owner_and_peer_channel_changes_require_distinct_blocks():
    duplicate = report([block(), block(tb_id=1)])
    assert 'connection_owner' in {v.code for v in duplicate.violations}
    from vericcl.xml.capacity import ExecutionShape, RuntimeLimits, check_capacity
    shape = ExecutionShape(3, 2, (block(), block(tb_id=1, send_peer=2), block(tb_id=2, channel=1)))
    assert check_capacity(shape, RuntimeLimits()).compatible


def test_rank_direction_and_channel_caps_use_the_actual_program():
    from vericcl.xml.capacity import ExecutionShape, RuntimeLimits, check_capacity
    blocks = tuple(block(tb_id=i, send_peer=i + 1) for i in range(33))
    report_ = check_capacity(ExecutionShape(34, 1, blocks), RuntimeLimits())
    assert 'send_tbs_per_channel' in {v.code for v in report_.violations}
    assert not report_.repairable_by_slice_only
    local = tuple(block(tb_id=i, channel=-1, send_peer=-1, recv_peer=-1, step_count=0) for i in range(217))
    assert 'tbs_per_rank' in {v.code for v in check_capacity(ExecutionShape(2, 1, local), RuntimeLimits()).violations}
    assert 'channels' in {v.code for v in check_capacity(ExecutionShape(2, 17, ()), RuntimeLimits()).violations}


def test_scratch_budget_and_dependency_targets_are_checked():
    from vericcl.xml.capacity import ExecutionShape, RuntimeLimits, StepCapacity, check_capacity
    shape = ExecutionShape(2, 1, (block(step_count=0, **{'steps': (StepCapacity('s', src_offset=0, dependency_tb=1, dependency_step=0),)}),),
                           scratch_slots={0: 2})
    result = check_capacity(shape, RuntimeLimits(), slice_bytes=1024, scratch_bytes_per_rank=1024)
    assert {'scratch_bytes', 'dependency_target'} <= {v.code for v in result.violations}


def test_buffer_capacity_uses_typed_step_geometry_and_allows_extra_space():
    from vericcl.xml.capacity import simple_payload_capacity
    from vericcl.xml.granularity import verify_atom_granularity
    assert simple_payload_capacity(2_097_153, 4) == 1_048_576
    verify_atom_granularity(256, 1, 4, 1, 1024, 16_777_216)
    with pytest.raises(SemanticError, match='signed 32'):
        simple_payload_capacity(4 * 1024**3, 4)


def config(tmp_path, *, operator='broadcast', ranks=2, size=4096, slice_size=None, steps=256, channels=None):
    docs = request_documents(operator, ranks)
    docs['sketch']['hyperparameters'] = {'total_size_bytes': size, 'slice_size_bytes': slice_size}
    docs['runtime']['limits'] = {'steps_per_tb': steps}
    if channels is not None:
        docs['sketch']['solver']['max_channels'] = channels
    return load_request(write_request(tmp_path, docs))


def test_initial_domain_uses_default_channels_and_one_legal_slice_per_channel(tmp_path):
    from vericcl.planner.parameters import initial_parameter_domain
    inputs = config(tmp_path)
    domain = initial_parameter_domain(inputs)
    assert [candidate.channels for candidate in domain.candidates] == list(range(1, 8))
    assert domain.candidates[0].slice_bytes == 16
    assert all(4096 % candidate.slice_bytes == 0 for candidate in domain.candidates)
    assert all(candidate.requires_structure_check for candidate in domain.candidates)
    assert not domain.complete_runtime_validation


def test_explicit_channel_extension_and_non_power_of_two_slices(tmp_path):
    from vericcl.planner.parameters import initial_parameter_domain
    domain = initial_parameter_domain(config(tmp_path, size=1200, steps=16, channels=8))
    assert domain.candidates[0].slice_bytes == 80
    assert domain.candidates[-1].channels == 8


def test_explicit_slice_is_retained_or_adjusted_only_for_capacity(tmp_path):
    from vericcl.planner.parameters import initial_parameter_domain
    retained = initial_parameter_domain(config(tmp_path, slice_size=1024))
    assert all(candidate.slice_bytes == 1024 for candidate in retained.candidates)
    adjusted = initial_parameter_domain(config(tmp_path, slice_size=4))
    assert adjusted.candidates[0].slice_bytes == 16
    assert adjusted.candidates[0].adjustment_reason == 'initial_slice_capacity'
    fixed = initial_parameter_domain(config(tmp_path, slice_size=4), fixed_slice_bytes=4)
    assert any(item.channels == 1 and item.proven for item in fixed.exclusions)


def test_reduce_scatter_and_all_to_all_keep_rank_divisibility(tmp_path):
    from vericcl.planner.parameters import initial_parameter_domain
    inputs = config(tmp_path, operator='reduce_scatter', ranks=4, size=4096)
    domain = initial_parameter_domain(inputs)
    assert all(candidate.slice_count % 4 == 0 for candidate in domain.candidates)
    with pytest.raises(InputValidationError):
        config(tmp_path, operator='alltoall', ranks=4, size=12)


def test_no_slice_size_can_fix_two_required_directions_in_one_separate_tb(tmp_path):
    from vericcl.planner.parameters import initial_parameter_domain
    docs = request_documents('allreduce', 2)
    docs['runtime']['limits'] = {'tbs_per_rank': 1}
    domain = initial_parameter_domain(load_request(write_request(tmp_path, docs)))
    assert not domain.candidates
    assert all(item.proven and item.code == 'structural_tb_minimum' for item in domain.exclusions)


def test_fused_layout_does_not_count_one_fused_step_twice(tmp_path):
    from vericcl.planner.parameters import initial_parameter_domain
    docs = request_documents('allreduce', 2)
    docs['runtime'].update(tb_layout='fused', limits={'tbs_per_rank': 1})
    docs['sketch']['hyperparameters'].update(total_size_bytes=4096, slice_size_bytes=16)
    candidate = initial_parameter_domain(load_request(write_request(tmp_path, docs))).candidates[0]
    assert candidate.slice_bytes == 16


def test_bound_attempt_records_effective_partition_without_mutating_request(tmp_path):
    from vericcl.planner.parameters import initial_parameter_domain, resolve_parameters
    original = config(tmp_path)
    candidate = initial_parameter_domain(original).candidates[0]
    bound = resolve_parameters(original, candidate)
    assert bound.hyperparameters.slice_size_bytes == 16
    assert bound.request_config['sketch']['hyperparameters']['slice_size_bytes'] == 16
    assert bound.request_config['sketch']['solver']['max_channels'] == 1
    assert bound.size_accounting['slices_per_rank'] == 256
    assert bound.value_sources['sketch.hyperparameters.slice_size_bytes']['kind'] == 'parameter_selection'
    assert original.request_config['sketch']['hyperparameters']['slice_size_bytes'] is None
    assert bound.input_sha256 == original.input_sha256


def test_xml_capacity_check_uses_limits_and_detects_duplicate_owners():
    from hashlib import sha256
    from copy import deepcopy
    from dataclasses import replace
    from lxml import etree
    from tests.unit.xml.test_compatibility import _artifact
    from vericcl.xml.compatibility import check_msccl_compatibility
    from vericcl.xml.capacity import RuntimeLimits
    artifact = _artifact()
    root = etree.fromstring(artifact.xml_text.encode())
    tb = root.xpath('./gpu/tb')[0]
    step = deepcopy(tb.xpath('./step')[0])
    step.attrib['s'] = '1'
    tb.append(step)
    xml = etree.tostring(root, encoding='unicode')
    longer = replace(artifact, xml_text=xml, sha256=sha256(xml.encode()).hexdigest())
    assert check_msccl_compatibility(longer).runtime_compatible
    limited = check_msccl_compatibility(longer, limits=RuntimeLimits(steps_per_tb=1))
    assert not limited.runtime_compatible
    root = etree.fromstring(artifact.xml_text.encode())
    gpu = root.xpath('./gpu')[0]
    duplicate = deepcopy(next(tb for tb in gpu.xpath('./tb') if int(tb.attrib['send']) >= 0))
    duplicate.attrib['id'] = str(len(gpu.xpath('./tb')))
    gpu.append(duplicate)
    xml = etree.tostring(root, encoding='unicode')
    changed = replace(artifact, xml_text=xml, sha256=sha256(xml.encode()).hexdigest())
    assert 'connection_owner' in {issue.code for issue in check_msccl_compatibility(changed).issues}


def test_peer_and_loop_fields_are_checked_against_storage_widths():
    from vericcl.xml.capacity import ExecutionShape, RuntimeLimits, check_capacity
    shape = ExecutionShape(32769, 1, (block(send_peer=32768),), chunks_per_loop=2**31)
    assert {'peer_id', 'chunks_per_loop'} <= {issue.code for issue in check_capacity(shape, RuntimeLimits()).violations}


def parser_blocks(rank=0, last_steps=236):
    from vericcl.xml.capacity import BlockCapacity, StepCapacity
    # 15*256+236 = 4076 steps; 16 TB headers are retained for this rank.
    return tuple(BlockCapacity(rank, i, -1, -1, -1,
        tuple(StepCapacity('nop') for _ in range(256 if i < 15 else last_steps))) for i in range(16))


def test_xml_parser_exact_boundary_counts_algo_all_gpu_headers_tb_and_nop():
    from vericcl.xml.capacity import ExecutionShape, RuntimeLimits, check_capacity
    assert check_capacity(ExecutionShape(2, 1, parser_blocks()), RuntimeLimits()).compatible
    rejected = check_capacity(ExecutionShape(2, 1, parser_blocks(last_steps=237)), RuntimeLimits())
    issues = [v for v in rejected.violations if v.code == 'xml_nodes']
    assert [(v.rank, v.tb_id, v.current, v.limit) for v in issues] == [(0, None, 4096, 4095)]
    # A third GPU header consumes one node even when that GPU has no TBs.
    extra_header = check_capacity(ExecutionShape(3, 1, parser_blocks()), RuntimeLimits())
    assert [(v.rank, v.current) for v in extra_header.violations if v.code == 'xml_nodes'] == [(0, 4096)]


def test_xml_parser_budget_is_per_rank_not_full_document():
    from vericcl.xml.capacity import ExecutionShape, RuntimeLimits, check_capacity
    shape = ExecutionShape(2, 1, parser_blocks(0) + parser_blocks(1))
    assert check_capacity(shape, RuntimeLimits()).compatible


def test_xml_parser_rejects_gpu_headers_that_fill_the_parser_array():
    from vericcl.xml.capacity import ExecutionShape, RuntimeLimits, check_capacity
    result = check_capacity(ExecutionShape(4095, 1, ()), RuntimeLimits())
    assert [(v.rank, v.current, v.limit) for v in result.violations] == [(None, 4096, 4095)]


def test_initial_slice_domain_includes_fixed_xml_parser_space(tmp_path):
    from vericcl.planner.parameters import initial_parameter_domain
    inputs = config(tmp_path, operator='allgather', ranks=16, size=4194304,
                    slice_size=16384, channels=2)
    domain = initial_parameter_domain(inputs)
    assert [candidate.slice_bytes for candidate in domain.candidates] == [32768, 32768]
    assert all(dict(candidate.bounds)['xml_nodes_rank_0'] == 254 for candidate in domain.candidates)
    fixed = initial_parameter_domain(inputs, fixed_slice_bytes=16384)
    assert not fixed.candidates
    assert all(item.code == 'fixed_slice_capacity' and item.proven for item in fixed.exclusions)


def test_one_rank_outofplace_copy_uses_xml_parser_space(tmp_path):
    from vericcl.planner.parameters import initial_parameter_domain
    inputs = config(tmp_path, operator='allreduce', ranks=1, size=65536, slice_size=4, channels=1)
    domain = initial_parameter_domain(inputs)
    assert dict(domain.candidates[0].bounds)['xml_nodes_rank_0'] == 4092
    assert domain.candidates[0].slice_bytes == 32
