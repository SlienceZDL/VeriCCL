"""Atoms of symbolic execution: {s, pd, t} (one model for the whole code base).

The types live in vericcl.semantics.atom; this module re-exports them for the
symbolic executor. Symbols are created only by symbolic execution.
"""
from vericcl.semantics.atom import (  # noqa: F401
    Atom, PathDAG, PathStage, SliceVec, Symbol, TimeVec, _executor_symbol, symbolic_step,
)


def _make_symbol(symbol_id, src_rank, dst_rank, ready_time, kind, channel, stage):
    return _executor_symbol(symbol_id, src_rank, dst_rank, ready_time, kind, channel, stage)
