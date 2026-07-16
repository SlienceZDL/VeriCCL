from enum import Enum

class Collective(Enum):
    AllReduce = 1
    ReduceScatter = 2
    AllGather = 3
    Reduce = 4
    Broadcast = 5
    Gather = 6