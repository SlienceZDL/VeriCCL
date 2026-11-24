"""Small CustomCollective programs for actual pinned-runtime primitives."""
from dataclasses import dataclass
from collections import Counter
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class ProbeArtifact:
    xml: str
    endpoints: tuple[dict, ...]


def build_probe_xml(job):
    mapping = {entry['original_rank']: entry['rank'] for entry in job.rank_map}
    steps = getattr(job, 'chain_steps', 1)
    chunks = len(job.flows) * steps
    if not chunks or chunks > 32768 or steps > 256:
        raise ValueError('probe needs representable independent payload slots')
    root = ET.Element('algo', name='VeriCCL-probe', proto='Simple', coll='custom',
                      ngpus=str(len(mapping)), nchunksperloop=str(chunks),
                      nchannels=str(max(f.channel for f in job.flows) + 1),
                      inplace='0', nthreads='256', minBytes='0', maxBytes=str(2**63 - 1))
    endpoints, channels = [], Counter()
    for original, rank in sorted(mapping.items(), key=lambda x: x[1]):
        gpu = ET.SubElement(root, 'gpu', id=str(rank), i_chunks=str(chunks), o_chunks=str(chunks), s_chunks='0')
        tb_id = 0
        for index, flow in enumerate(job.flows):
            if original not in (flow.src_rank, flow.dst_rank):
                continue
            local = flow.src_rank == flow.dst_rank
            role = flow.kind if local or original == flow.dst_rank else 'send'
            if flow.kind == 'send' and original == flow.dst_rank:
                role = 'recv'
            send = mapping[flow.dst_rank] if role == 'send' else -1
            recv = mapping[flow.src_rank] if role in {'recv', 'rrc'} else -1
            native = {'send': 's', 'recv': 'r', 'rrc': 'rrc', 'copy': 'cpy', 'reduce': 're'}[role]
            tb = ET.SubElement(gpu, 'tb', id=str(tb_id), send=str(send), recv=str(recv), chan=str(flow.channel))
            # A flow's buffer slot holds ``steps`` consecutive chunks; its
            # steps move them in order on one TB (a pipelined chain).
            for step in range(steps):
                chunk = index * steps + step
                ET.SubElement(tb, 'step', s=str(step), type=native, srcbuf='i', srcoff=str(chunk),
                              dstbuf='o', dstoff=str(chunk), cnt='1', depid='-1', deps='-1', hasdep='0')
                endpoints.append(dict(rank=rank, original_rank=original, tb=tb_id, step=step,
                                      flow_id=flow.flow_id, role=role, channel=flow.channel, slot=index))
            if not local:
                key = (rank, flow.channel, 'send' if send >= 0 else 'recv')
                channels[key] += 1
                if channels[key] > 32:
                    raise ValueError('probe exceeds per-channel endpoint TB capacity')
            tb_id += 1
        if not 0 < tb_id <= 216:
            raise ValueError('probe has an empty rank or exceeds TB capacity')
    ET.indent(root)
    return ProbeArtifact(ET.tostring(root, encoding='unicode') + '\n', tuple(endpoints))
