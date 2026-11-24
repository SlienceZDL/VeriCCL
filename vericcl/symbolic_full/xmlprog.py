"""Independent reading of an MSCCL XML program (no lowering objects involved)."""
from __future__ import annotations

from dataclasses import dataclass
import xml.etree.ElementTree as ET
from typing import Dict, Tuple

from vericcl.errors import SemanticError


@dataclass(frozen=True)
class XStep:
    rank: int
    tb: int
    index: int
    kind: str
    srcbuf: str
    srcoff: int
    dstbuf: str
    dstoff: int
    count: int
    depid: int
    deps: int
    hasdep: bool


@dataclass(frozen=True)
class XBlock:
    rank: int
    tb: int
    send: int
    recv: int
    channel: int
    steps: Tuple[XStep, ...]


@dataclass(frozen=True)
class XProgram:
    ngpus: int
    chunks_per_loop: int
    channels: int
    inplace: bool
    collective: str
    min_bytes: int
    max_bytes: int
    blocks: Tuple[XBlock, ...]
    buffers: Dict[int, Tuple[int, int, int]]          # rank -> (i_chunks, o_chunks, s_chunks)


def _int(element, name, default=None):
    value = element.get(name)
    if value is None:
        if default is None:
            raise SemanticError('XML element <%s> lacks %s' % (element.tag, name))
        return default
    try:
        return int(value)
    except ValueError as error:
        raise SemanticError('XML attribute %s=%r is not an integer' % (name, value)) from error


def parse_xml(text: str) -> XProgram:
    try:
        root = ET.fromstring(text)
    except ET.ParseError as error:
        raise SemanticError('XML does not parse: %s' % error) from error
    if root.tag != 'algo':
        raise SemanticError('XML root must be <algo>')
    blocks, buffers = [], {}
    for gpu in root.findall('gpu'):
        rank = _int(gpu, 'id')
        buffers[rank] = (_int(gpu, 'i_chunks', 0), _int(gpu, 'o_chunks', 0), _int(gpu, 's_chunks', 0))
        for tb in gpu.findall('tb'):
            tb_id = _int(tb, 'id')
            steps = []
            for position, step in enumerate(tb.findall('step')):
                if _int(step, 's') != position:
                    raise SemanticError('XML steps of rank %d tb %d are not numbered 0..n-1' % (rank, tb_id))
                steps.append(XStep(rank, tb_id, position, step.get('type'), step.get('srcbuf'), _int(step, 'srcoff'),
                                   step.get('dstbuf'), _int(step, 'dstoff'), _int(step, 'cnt', 1), _int(step, 'depid', -1),
                                   _int(step, 'deps', -1), bool(_int(step, 'hasdep', 0))))
            blocks.append(XBlock(rank, tb_id, _int(tb, 'send'), _int(tb, 'recv'), _int(tb, 'chan', 0), tuple(steps)))
    return XProgram(_int(root, 'ngpus'), _int(root, 'nchunksperloop'), _int(root, 'nchannels', 1),
                    root.get('inplace') == '1', root.get('coll', ''), _int(root, 'minBytes', 0),
                    _int(root, 'maxBytes', 0), tuple(blocks), buffers)
