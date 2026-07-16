from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CollectiveKind(str, Enum):
    BROADCAST = "broadcast"
    REDUCE = "reduce"
    ALL_GATHER = "allgather"
    ALL_REDUCE = "allreduce"
    ALL_TO_ALL = "alltoall"
    REDUCE_SCATTER = "reduce_scatter"


@dataclass(frozen=True)
class CollectiveSpec:
    kind: CollectiveKind
    datatype: str
    reduction_op: Optional[str] = None
    root: Optional[int] = None
    inplace: bool = False
