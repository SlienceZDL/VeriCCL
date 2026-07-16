from enum import Enum

class Slice(Enum):
    Rack = 1
    Server = 2
    CPU = 3
    # GPU = 4

class Form(Enum):
    InsideGroup = 1
    Parallel_e = 2
    Master_e = 3

class Collective(Enum):
    AllReduce = 1
    ReduceScatter = 2
    AllGather = 3
    Reduce = 4
    Broadcast = 5