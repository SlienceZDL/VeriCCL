"""Threadblock-order hints: Inverted & ~Necessary over real threadblock steps.

Inverted(step_i, step_j): both steps run on one threadblock, i comes first,
j is the later step whose deciding wait is threadblock order, and j's data is
ready before i's data (so j idles behind i).
Necessary(step_i, step_j): j depends on i through data and dependency flags
alone - the BDD closure of the waits without threadblock order - so swapping
them cannot help.
A hint is a pair in Inverted & ~Necessary; it becomes an ``order`` repair
strategy that runs j before i, ranked by the time j would gain and by whether
j lies on the critical chain.
"""
from __future__ import annotations

from vericcl.verification.bottlenecks import StrategyRecord

from .config import settings as read_settings
from .view import critical_chain

WINDOW = 8


def tb_order_hints(execution, inputs, *, lease=None):
    """[(StrategyRecord, gain_us)] after ``execution.bind``."""
    full = read_settings(inputs)
    s = execution.space
    times = execution.times()
    ready = {index: execution.value(execution.data_ready[index]) for index in execution.order}
    tail = max(execution.order, key=lambda index: (times[index][1], index)) if execution.order else None
    critical = set(critical_chain(execution, tail)) if tail is not None else set()
    rows, gains = set(), {}
    for members in execution.blocks.values():
        steps = [index for index in members if execution.nodes[index].kind != 'nop' and execution.nodes[index].operation_id]
        for position, j in enumerate(steps):
            if lease is not None and position % 64 == 0:
                lease.check()
            label, _ = execution.critical_predecessor(j)
            if label != 'tb':
                continue
            for i in steps[max(0, position - WINDOW):position]:
                gain = ready[i] - ready[j]
                if gain > 0 and execution.nodes[i].operation_id != execution.nodes[j].operation_id:
                    rows.add((i, j))
                    gains[i, j] = gain
    if not rows:
        return []
    inverted = s.relation(('node', 'node2'), rows)
    # Necessary pairs are ordered by data and dependency waits alone; hints are the rest.
    hints = execution.unordered(inverted, with_tb=False)
    pairs = s.tuples(hints, ('node', 'node2'))
    ranked = sorted(pairs, key=lambda pair: (pair[1] not in critical, -gains[pair], pair))
    result, seen = [], set()
    for i, j in ranked:
        first, second = execution.nodes[i].operation_id, execution.nodes[j].operation_id
        if (first, second) in seen:
            continue
        seen.add((first, second))
        result.append((StrategyRecord('order', (first, second), orders=((second, first),),
                                      cause_ids=('tb_order:%s:%s' % (first, second),)), gains[i, j]))
        if len(result) >= full.max_order_hints:
            break
    return result
