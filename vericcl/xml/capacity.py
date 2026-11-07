"""Typed capacity checks for the concrete Simple/cnt=1 execution program."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping

from vericcl.errors import SemanticError


# Pinned MSCCL xml.cc checks maxIndex before reading a closing tag or EOF.
# Its 4096-entry array therefore admits at most 4095 retained XML nodes.
# Each rank retains algo + every GPU header + its own TB and step nodes.
XML_MAX_RETAINED_NODES = 4095


@dataclass(frozen=True)
class RuntimeLimits:
    steps_per_tb: int = 256
    tbs_per_rank: int = 216
    send_tbs_per_channel: int = 32
    recv_tbs_per_channel: int = 32
    dependency_id: int = 127
    buffer_offset: int = 32767
    max_channels: int = 16

    def __post_init__(self):
        maxima = {'steps_per_tb': 256, 'tbs_per_rank': 216, 'send_tbs_per_channel': 32,
                  'recv_tbs_per_channel': 32, 'dependency_id': 127, 'buffer_offset': 32767, 'max_channels': 16}
        for name, maximum in maxima.items():
            value = getattr(self, name)
            minimum = 0 if name in {'dependency_id', 'buffer_offset'} else 1
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise SemanticError(name + ' is outside the supported runtime limit')

    @classmethod
    def from_runtime(cls, runtime):
        return cls(**runtime.get('limits', {}))


def simple_payload_capacity(buffsize_bytes, datatype_bytes):
    if isinstance(buffsize_bytes, bool) or not isinstance(buffsize_bytes, int) or not 8 <= buffsize_bytes <= 2**31 - 1:
        raise SemanticError('NCCL_BUFFSIZE must be a signed 32-bit byte count of at least 8')
    if datatype_bytes not in (1, 2, 4, 8) or isinstance(datatype_bytes, bool):
        raise SemanticError('unsupported datatype byte width')
    # Pinned runtime: NCCL_STEPS=8, Simple CHUNKSTEPS=SLICESTEPS=4.
    # Mirror the device integer divisions; a rounded-up B/2 is not sufficient.
    return ((buffsize_bytes // 8) // datatype_bytes) * datatype_bytes * 4


# Pinned MSCCL computeColl (src/enqueue.cc) rejects a Simple-protocol launch
# unless its chunk, (NCCL_BUFFSIZE / NCCL_STEPS) * MSCCL_CHUNKSTEPS bytes, is a
# multiple of (nThreads - WARP_SIZE) * sizeof(uint64_t) / sizeof(datatype).
# VeriCCL XML leaves nthreads unset, so nThreads - WARP_SIZE is
# NCCL_SIMPLE_MAX_NTHREADS = 512. The native executor admits plans as custom
# collectives with their own datatype; the legacy NCCL AllGather/Broadcast API
# path counts bytes (datatype width 1) and is therefore stricter.
MSCCL_SIMPLE_COPY_THREADS = 512


def simple_chunk_granule(datatype_bytes):
    if datatype_bytes not in (1, 2, 4, 8) or isinstance(datatype_bytes, bool):
        raise SemanticError('unsupported datatype byte width')
    return MSCCL_SIMPLE_COPY_THREADS * 8 // datatype_bytes


def simple_chunk_geometry(buffsize_bytes, datatype_bytes):
    """Return (chunk_bytes, required_multiple, valid) for the pinned runtime."""
    simple_payload_capacity(buffsize_bytes, datatype_bytes)
    chunk = (buffsize_bytes // 8) * 4
    granule = simple_chunk_granule(datatype_bytes)
    return chunk, granule, chunk % granule == 0


@dataclass(frozen=True)
class StepCapacity:
    kind: str
    src_offset: int = -1
    dst_offset: int = -1
    dependency_tb: int = -1
    dependency_step: int = -1
    count: int = 1


@dataclass(frozen=True)
class BlockCapacity:
    rank: int
    tb_id: int
    channel: int
    send_peer: int
    recv_peer: int
    steps: tuple[StepCapacity, ...]


@dataclass(frozen=True)
class ExecutionShape:
    rank_count: int
    channels: int
    blocks: tuple[BlockCapacity, ...]
    scratch_slots: Mapping[int, int] = field(default_factory=dict)
    chunks_per_loop: int | None = None


@dataclass(frozen=True)
class CapacityViolation:
    code: str
    rank: int | None
    tb_id: int | None
    current: int
    limit: int
    detail: str = ''


@dataclass(frozen=True)
class CapacityReport:
    violations: tuple[CapacityViolation, ...]

    @property
    def compatible(self):
        return not self.violations

    @property
    def repairable_by_slice_only(self):
        # These are size-sensitive limits, not a promise that any resized
        # program is feasible. Other violations require structural changes.
        return bool(self.violations) and all(issue.code in {'steps_per_tb', 'buffer_offset', 'chunks_per_loop'} for issue in self.violations)


def check_capacity(shape, limits, *, slice_bytes=None, scratch_bytes_per_rank=None):
    if not isinstance(shape, ExecutionShape) or not isinstance(limits, RuntimeLimits):
        raise SemanticError('capacity check needs an execution shape and runtime limits')
    if isinstance(shape.rank_count, bool) or not isinstance(shape.rank_count, int) or shape.rank_count < 1:
        raise SemanticError('execution shape needs a positive rank count')
    issues = []
    def issue(code, block, current, limit, detail=''):
        issues.append(CapacityViolation(code, None if block is None else block.rank,
                                        None if block is None else block.tb_id, current, limit, detail))
    if isinstance(shape.channels, bool) or not isinstance(shape.channels, int) or not 1 <= shape.channels <= limits.max_channels:
        issue('channels', None, shape.channels, limits.max_channels)
    if shape.chunks_per_loop is not None and (isinstance(shape.chunks_per_loop, bool) or not isinstance(shape.chunks_per_loop, int)
                                             or not 1 <= shape.chunks_per_loop <= 2**31 - 1):
        issue('chunks_per_loop', None, shape.chunks_per_loop, 2**31 - 1)
    by_id, rank_blocks, directions, owners = {}, Counter(), Counter(), set()
    rank_xml_nodes = Counter()
    for block in shape.blocks:
        if not isinstance(block, BlockCapacity):
            raise SemanticError('capacity block type is invalid')
        for name in ('rank', 'tb_id', 'channel', 'send_peer', 'recv_peer'):
            value = getattr(block, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise SemanticError('block ' + name + ' must be an integer')
        if not 0 <= block.rank < shape.rank_count:
            issue('rank_id', block, block.rank, shape.rank_count - 1)
        if not 0 <= block.tb_id < limits.tbs_per_rank:
            issue('tb_id', block, block.tb_id, limits.tbs_per_rank - 1)
        key = block.rank, block.tb_id
        if key in by_id:
            issue('duplicate_tb_id', block, block.tb_id, limits.tbs_per_rank - 1)
        by_id[key] = block
        rank_blocks[block.rank] += 1
        rank_xml_nodes[block.rank] += 1 + len(block.steps)
        local = block.send_peer == block.recv_peer == -1
        if not ((local and block.channel == -1) or 0 <= block.channel < shape.channels):
            issue('channel_id', block, block.channel, shape.channels - 1)
        if len(block.steps) > limits.steps_per_tb:
            issue('steps_per_tb', block, len(block.steps), limits.steps_per_tb)
        for direction in ('send', 'recv'):
            peer = getattr(block, direction + '_peer')
            if peer == -1:
                continue
            if not 0 <= peer <= min(32767, shape.rank_count - 1) or peer == block.rank:
                issue('peer_id', block, peer, min(32767, shape.rank_count - 1))
            directions[block.rank, block.channel, direction] += 1
            owner = block.rank, peer, block.channel, direction
            if owner in owners:
                issue('connection_owner', block, 2, 1, direction)
            owners.add(owner)
        for step in block.steps:
            if not isinstance(step, StepCapacity):
                raise SemanticError('capacity step type is invalid')
            for name in ('src_offset', 'dst_offset', 'dependency_tb', 'dependency_step', 'count'):
                if isinstance(getattr(step, name), bool) or not isinstance(getattr(step, name), int):
                    raise SemanticError('step ' + name + ' must be an integer')
            if step.count != 1:
                issue('runtime_count', block, step.count, 1)
            for offset in (step.src_offset, step.dst_offset):
                if not -1 <= offset <= limits.buffer_offset:
                    issue('buffer_offset', block, offset, limits.buffer_offset)
            if not -1 <= step.dependency_tb <= limits.dependency_id:
                issue('dependency_id', block, step.dependency_tb, limits.dependency_id)
            if not -1 <= step.dependency_step < limits.steps_per_tb:
                issue('dependency_step', block, step.dependency_step, limits.steps_per_tb - 1)
    header_nodes = 1 + shape.rank_count
    if header_nodes > XML_MAX_RETAINED_NODES:
        issue('xml_nodes', None, header_nodes, XML_MAX_RETAINED_NODES,
              'algo and all GPU headers alone exceed the fixed parser capacity')
    else:
        for rank, nodes in rank_xml_nodes.items():
            if header_nodes + nodes > XML_MAX_RETAINED_NODES:
                issues.append(CapacityViolation('xml_nodes', rank, None,
                    header_nodes + nodes, XML_MAX_RETAINED_NODES,
                    'algo + all GPU headers + this rank TB headers and all steps, including NOP'))
    for rank, count in rank_blocks.items():
        if count > limits.tbs_per_rank:
            issues.append(CapacityViolation('tbs_per_rank', rank, None, count, limits.tbs_per_rank))
        ids = {block.tb_id for block in shape.blocks if block.rank == rank}
        if ids != set(range(count)):
            issues.append(CapacityViolation('dense_tb_ids', rank, None, max(ids), count - 1))
    for (rank, channel, direction), count in directions.items():
        maximum = getattr(limits, direction + '_tbs_per_channel')
        if count > maximum:
            issues.append(CapacityViolation(direction + '_tbs_per_channel', rank, None, count, maximum, 'channel=' + str(channel)))
    for block in shape.blocks:
        for step in block.steps:
            target = by_id.get((block.rank, step.dependency_tb))
            if step.dependency_tb == -1 and step.dependency_step == -1:
                continue
            if target is None or not 0 <= step.dependency_step < len(target.steps):
                issue('dependency_target', block, step.dependency_tb, limits.dependency_id)
    if slice_bytes is not None and scratch_bytes_per_rank is not None:
        if isinstance(slice_bytes, bool) or not isinstance(slice_bytes, int) or slice_bytes < 1:
            raise SemanticError('scratch accounting needs a positive slice size')
        for rank, slots in shape.scratch_slots.items():
            budget = scratch_bytes_per_rank.get(rank, 0) if isinstance(scratch_bytes_per_rank, Mapping) else scratch_bytes_per_rank
            if (isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < shape.rank_count
                    or isinstance(slots, bool) or not isinstance(slots, int) or slots < 0
                    or isinstance(budget, bool) or not isinstance(budget, int) or budget < 0):
                raise SemanticError('invalid scratch allocation or budget')
            if slots * slice_bytes > budget:
                issues.append(CapacityViolation('scratch_bytes', rank, None, slots * slice_bytes, budget))
    return CapacityReport(tuple(issues))
