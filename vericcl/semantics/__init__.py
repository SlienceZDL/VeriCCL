"""Collective semantics and immutable schedule state."""

from vericcl.semantics.atom import Atom, PathStage, Schedule, Symbol, Transfer
from vericcl.semantics.checker import check_final_states
from vericcl.semantics.collective import (
    CollectiveKind,
    CollectiveSpec,
    OutputSlot,
    required_outputs,
)
from vericcl.semantics.slice import logical_slice_index, source_rank
from vericcl.semantics.ranges import DataRange, ConstraintSet, bind_constraints
from vericcl.semantics.versions import BufferRegion, StateVersion, VersionLedger
from vericcl.semantics.version_schedule import OutputBinding, VersionOperation, VersionSchedule
from vericcl.semantics.state import (
    PayloadLedger,
    PayloadState,
    initial_payload_states,
)

__all__ = [
    "DataRange", "ConstraintSet", "bind_constraints", "BufferRegion", "StateVersion", "VersionLedger",
    "OutputBinding", "VersionOperation", "VersionSchedule",
    "Atom",
    "CollectiveKind",
    "CollectiveSpec",
    "OutputSlot",
    "PathStage",
    "PayloadLedger",
    "PayloadState",
    "Schedule",
    "Symbol",
    "Transfer",
    "check_final_states",
    "initial_payload_states",
    "logical_slice_index",
    "required_outputs",
    "source_rank",
]
