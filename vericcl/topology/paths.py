import heapq
import math
from typing import Dict, Set, Tuple

from vericcl.errors import SemanticError
from vericcl.topology.model import LinkKey, Topology


def _rank(topology: Topology, value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SemanticError("{} rank must be an integer".format(field))
    if value < 0 or value >= topology.rank_count:
        raise SemanticError("{} rank is outside the topology".format(field))
    return value


def _equal(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def shortest_path_set(
    topology: Topology,
    src: int,
    dst: int,
) -> Tuple[Tuple[int, ...], ...]:
    if not isinstance(topology, Topology):
        raise SemanticError("topology must be a Topology")
    source = _rank(topology, src, "source")
    destination = _rank(topology, dst, "destination")
    if source == destination:
        return ((source,),)

    distances = {rank: math.inf for rank in range(topology.rank_count)}
    predecessors: Dict[int, Set[int]] = {
        rank: set() for rank in range(topology.rank_count)
    }
    distances[source] = 0.0
    pending = [(0.0, source)]
    while pending:
        distance, rank = heapq.heappop(pending)
        if distance > distances[rank] and not _equal(distance, distances[rank]):
            continue
        for next_rank in topology.destinations(rank):
            edge = topology.link(LinkKey(rank, next_rank))
            candidate = distance + edge.performance.invbw_us
            if candidate < distances[next_rank] and not _equal(
                candidate,
                distances[next_rank],
            ):
                distances[next_rank] = candidate
                predecessors[next_rank] = {rank}
                heapq.heappush(pending, (candidate, next_rank))
            elif _equal(candidate, distances[next_rank]):
                predecessors[next_rank].add(rank)

    if math.isinf(distances[destination]):
        return ()

    def enumerate_paths(
        rank: int,
        visited: frozenset,
    ) -> Tuple[Tuple[int, ...], ...]:
        if rank == source:
            return ((source,),)
        paths = []
        for predecessor in sorted(predecessors[rank]):
            if predecessor in visited:
                continue
            for prefix in enumerate_paths(
                predecessor,
                visited | frozenset((predecessor,)),
            ):
                paths.append(prefix + (rank,))
        return tuple(paths)

    return tuple(
        sorted(
            set(
                enumerate_paths(
                    destination,
                    frozenset((destination,)),
                )
            )
        )
    )
