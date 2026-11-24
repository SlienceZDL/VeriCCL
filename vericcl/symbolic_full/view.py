"""Operation-level view of a bound symbolic execution (one entry per logical transfer)."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from vericcl.topology.model import LinkKey

from .executor import RECV_KINDS


@dataclass
class OpView:
    operation_id: str
    kind: str
    src_rank: int
    dst_rank: int
    node: int                       # the receiving (or copying) step
    send_node: Optional[int]
    start: float
    end: float
    data_ready: float
    bytes: int
    resources: Tuple[str, ...]
    contributors: frozenset
    stage: int
    channel: int


def resources_for(execution, operation_id, src, dst, topology):
    known = execution.link.resources.get(operation_id)
    if known:
        return tuple(dict.fromkeys(known))
    if src == dst:
        return ()
    edge = topology.links.get(LinkKey(src, dst))
    return ('link:%d:%d' % (src, dst),) + (tuple(edge.resource_ids) if edge is not None else ())


def operation_view(execution, topology) -> Dict[str, OpView]:
    """Requires ``execution.bind(...)`` first."""
    times = execution.times()
    contributors = defaultdict(set)
    first = {}
    for transfer in execution.transfers:
        key = transfer.operation_id or 'n%d' % transfer.recv
        contributors[key].update(execution.space.values(transfer.contrib, 'slice'))
        first.setdefault(key, transfer)
    view = {}
    for key, transfer in first.items():
        node = execution.nodes[transfer.recv]
        start, end = times[transfer.recv]
        view[key] = OpView(key, 'rrc' if node.kind in {'rrc', 'rrs', 'rrcs'} else 'send', transfer.src_rank,
                           transfer.dst_rank, transfer.recv, transfer.send, start, end,
                           execution.value(execution.data_ready[transfer.recv]),
                           execution.slice_bytes * max(1, node.count),
                           resources_for(execution, key, transfer.src_rank, transfer.dst_rank, topology),
                           frozenset(contributors[key]), transfer.stage, transfer.channel)
    for node in execution.nodes:
        if node.kind == 'cpy' and node.index in times and node.operation_id and node.operation_id not in view:
            start, end = times[node.index]
            view[node.operation_id] = OpView(node.operation_id, 'copy', node.rank, node.rank, node.index, None, start, end,
                                             execution.value(execution.data_ready[node.index]),
                                             execution.slice_bytes * max(1, node.count), (), frozenset(), node.stage, 0)
    return view


def op_of_node(execution, index):
    node = execution.nodes[index]
    if node.operation_id:
        return node.operation_id
    return None


def operation_edges(execution):
    """Op-level waits (pred op, op) from the node-level progress model."""
    edges = set()
    for index, preds in execution.preds.items():
        target = op_of_node(execution, index)
        if target is None:
            continue
        for pred in preds:
            source = op_of_node(execution, pred)
            if source is not None and source != target:
                edges.add((source, target))
    return edges


def critical_chain(execution, index, limit=4096):
    """Nodes on the chain of critical predecessors ending at ``index`` (tail first)."""
    chain, current = [], index
    seen = set()
    while current is not None and current not in seen and len(chain) < limit:
        seen.add(current)
        chain.append(current)
        _, current = execution.critical_predecessor(current)
    return chain
