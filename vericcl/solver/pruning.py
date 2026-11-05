import heapq
from typing import Callable, FrozenSet, Iterable, Tuple

from vericcl.topology.model import LinkKey


Path = Tuple[int, ...]


def retain_shortest_paths(
    paths: Iterable[Path],
    edge_cost: Callable[[int, int], float],
) -> Tuple[Path, ...]:
    scored = tuple(
        (
            sum(edge_cost(src, dst) for src, dst in zip(path, path[1:])),
            path,
        )
        for path in paths
    )
    if not scored:
        return ()
    minimum = min(score for score, _ in scored)
    tolerance = max(1e-12, abs(minimum) * 1e-12)
    return tuple(
        path
        for score, path in scored
        if abs(score - minimum) <= tolerance
    )


def viable_path_links(
    links: FrozenSet[LinkKey],
    source: int,
    destination: int,
) -> FrozenSet[LinkKey]:
    forward = {source}
    reverse = {destination}
    changed = True
    while changed:
        changed = False
        for link in links:
            if link.src_rank in forward and link.dst_rank not in forward:
                forward.add(link.dst_rank)
                changed = True
            if link.dst_rank in reverse and link.src_rank not in reverse:
                reverse.add(link.src_rank)
                changed = True
    return frozenset(
        link
        for link in links
        if link.src_rank in forward
        and link.dst_rank in reverse
        and link.dst_rank != source
        and link.src_rank != destination
    )


def ranked_simple_paths(
    links: FrozenSet[LinkKey],
    source: int,
    destination: int,
    edge_cost: Callable[[int, int], float],
    limit: int = 32,
) -> Tuple[Path, ...]:
    destinations = {}
    for link in links:
        destinations.setdefault(link.src_rank, []).append(link.dst_rank)
    for values in destinations.values():
        values.sort()
    pending = [(0.0, 0, (source,))]
    paths = []
    while pending and len(paths) < limit:
        cost, hop_count, path = heapq.heappop(pending)
        rank = path[-1]
        if rank == destination:
            paths.append(path)
            continue
        for next_rank in destinations.get(rank, ()):
            if next_rank in path:
                continue
            heapq.heappush(
                pending,
                (
                    cost + edge_cost(rank, next_rank),
                    hop_count + 1,
                    path + (next_rank,),
                ),
            )
    return tuple(paths)
