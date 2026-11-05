"""Bind checked schedule geometry and external semantics to native execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def export_manifest(bundle, inputs, xml_path, *, trace_records=None, trace_prefix=None, light_telemetry=True, live_trace=False):
    if not bundle.can_execute or bundle.program is None or not bundle.xml_text:
        raise ValueError('native execution requires a fully qualified candidate')
    digest = hashlib.sha256(bundle.xml_text.encode()).hexdigest()
    if digest != bundle.xml_sha256:
        raise ValueError('checked XML identity has changed')
    program = bundle.program
    size = inputs.hyperparameters.total_size_bytes
    slice_bytes = inputs.hyperparameters.slice_size_bytes
    collective = inputs.collective
    contract = dict(collective=collective.kind.value, datatype=collective.datatype,
                    reduction_op=collective.reduction_op, root=collective.root,
                    inplace=collective.inplace, rank_count=inputs.rank_count, input_size_bytes=size)
    ranks = []
    for rank in range(inputs.rank_count):
        blocks = [block for block in program.blocks if block.rank == rank]
        # MSCCL folds dependency-only NOPs into the next transfer. Counter and
        # trace positions refer to that compacted runtime transfer array.
        native_steps = [dict(tb_id=block.tb_id, step_index=index, kind=step.kind,
                            channel=block.channel, send_peer=block.send_peer, recv_peer=block.recv_peer,
                            operation_id=step.operation_id, payload_bytes=slice_bytes)
                        for block in blocks
                        for index, step in enumerate(step for step in block.steps if step.kind != 'nop')]
        steps = len(native_steps)
        capacity = 0 if trace_records is None else trace_records.get(rank)
        if trace_records is not None and (type(capacity) is not int or capacity < max(1, steps) or trace_prefix is None):
            raise ValueError('trace capacity must cover one complete invocation per rank')
        input_offset = rank * size if collective.inplace and collective.kind.value == 'allgather' else 0
        output_offset = rank * (size // inputs.rank_count) if collective.inplace and collective.kind.value == 'reduce_scatter' else 0
        ranks.append(dict(rank=rank, input_bytes=program.input_slots[rank] * slice_bytes,
            output_bytes=program.output_slots[rank] * slice_bytes,
            scratch_bytes=program.scratch_slots[rank] * slice_bytes,
            input_offset_bytes=input_offset, output_offset_bytes=output_offset,
            input_read_only=not any(step.dst is not None and step.dst.buffer == 'i' for block in blocks for step in block.steps),
            threadblocks=len(blocks), expected_steps=steps, trace_records=capacity,
            steps=native_steps))
    return dict(schema_version=1, version_id=digest, xml_sha256=digest,
                xml_path=str(Path(xml_path).resolve()), contract_digest=_digest(contract), **contract,
                slice_bytes=slice_bytes, runtime_bytes=program.chunks_per_loop * slice_bytes,
                channels=program.channels, ranks=ranks, light_telemetry=bool(light_telemetry and (trace_records is None or live_trace)),
                trace_prefix='' if trace_prefix is None else str(Path(trace_prefix).resolve()),
                tolerances=dict(inputs.request_config['sketch']['tolerances']))
