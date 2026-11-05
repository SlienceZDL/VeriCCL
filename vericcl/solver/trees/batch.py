"""Batched rooted arborescences with exact integer residual-cut checks.

The extension rule follows the published batched branching construction
(see docs/architecture.md). This implementation uses explicit slice
identities, integer capacities and the shared task lease. Network facilities
must already be represented as rank links; extra resource constraints use the
integer column master rather than this edge-only theorem.
"""
from collections import deque
from dataclasses import dataclass, replace
from types import MappingProxyType

import networkx as nx

from vericcl.errors import ConstructionInfeasibleError, SemanticError


def _check(lease):
    if lease is not None:
        lease.check()


def _graph(nodes, capacities):
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for (src, dst), capacity in capacities.items():
        if src not in graph or dst not in graph or src == dst:
            raise SemanticError('tree capacity edge must join distinct declared nodes')
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
            raise SemanticError('tree capacities must be nonnegative integers')
        if capacity:
            graph.add_edge(src, dst, capacity=capacity)
    return graph


def rooted_cut_feasible(nodes, capacities, root_counts, lease=None):
    """The augmented-root min-cut condition for all-node out-tree packing."""
    graph = _graph(nodes, capacities)
    total = sum(root_counts.values())
    source = object()
    graph.add_node(source)
    for root, count in root_counts.items():
        if root not in nodes or isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise SemanticError('root counts must be positive integers on declared nodes')
        graph.add_edge(source, root, capacity=count)
    for target in nodes:
        _check(lease)
        if nx.maximum_flow_value(graph, source, target) < total:
            return False
    return True


@dataclass(frozen=True)
class TreeBatch:
    batch_id: int
    root: int
    items: tuple
    edges: tuple = ()
    vertices: frozenset = frozenset()

    @property
    def multiplicity(self):
        return len(self.items)


@dataclass(frozen=True)
class Extension:
    amount: int
    remainder_id: int | None = None


class BatchPacker:
    def __init__(self, nodes, capacities, root_items, lease=None):
        self.nodes = tuple(nodes)
        if not self.nodes or len(set(self.nodes)) != len(self.nodes):
            raise SemanticError('tree nodes must be nonempty and unique')
        self.lease = lease
        self.capacities = dict(capacities)
        root_items = {root: tuple(items) for root, items in root_items.items() if items}
        all_items = [item for items in root_items.values() for item in items]
        if not all_items or len(all_items) != len(set(all_items)):
            raise SemanticError('tree items must be nonempty and unique across roots')
        if not rooted_cut_feasible(self.nodes, self.capacities, {r: len(v) for r, v in root_items.items()}, lease):
            raise ConstructionInfeasibleError('rooted cut cannot carry all integer tree copies')
        self._batches = {i: TreeBatch(i, root, items, (), frozenset({root}))
                         for i, (root, items) in enumerate(sorted(root_items.items()))}
        self._active = set(self._batches)
        self.original_items = frozenset(all_items)

    @property
    def batches(self):
        return MappingProxyType(self._batches)

    def extension_amount(self, batch_id, src, dst):
        _check(self.lease)
        batch = self._batches[batch_id]
        capacity = self.capacities.get((src, dst), 0)
        if batch_id not in self._active or src not in batch.vertices or dst in batch.vertices or not capacity:
            return 0
        graph = _graph(self.nodes, self.capacities)
        # A finite integer larger than every possible finite cut replaces the
        # conceptual infinity on each active batch-source -> reached-node arc.
        infinity = sum(self.capacities.values()) + sum(self._batches[i].multiplicity for i in self._active) + 1
        others = 0
        for index in sorted(self._active):
            item = self._batches[index]
            source = ('batch', index)
            for vertex in item.vertices:
                graph.add_edge(source, vertex, capacity=infinity)
            if index != batch_id:
                graph.add_edge(src, source, capacity=item.multiplicity)
                others += item.multiplicity
        feasible = nx.maximum_flow_value(graph, src, dst) - others
        if feasible < 0:
            raise SemanticError('residual batch cut invariant was violated')
        return min(batch.multiplicity, capacity, feasible)

    def extend(self, batch_id, src, dst):
        amount = self.extension_amount(batch_id, src, dst)
        if not amount:
            return Extension(0)
        batch = self._batches[batch_id]
        remainder_id = None
        if amount < batch.multiplicity:
            remainder_id = len(self._batches)
            self._batches[remainder_id] = replace(batch, batch_id=remainder_id, items=batch.items[amount:])
            self._active.add(remainder_id)
        self._batches[batch_id] = replace(batch, items=batch.items[:amount],
                                         edges=batch.edges + ((src, dst),), vertices=batch.vertices | {dst})
        self.capacities[src, dst] -= amount
        return Extension(amount, remainder_id)

    def finish(self):
        queue = deque(sorted(self._active))
        while queue:
            index = queue.popleft()
            _check(self.lease)
            frontier = deque(sorted(self._batches[index].vertices))
            while frontier:
                src = frontier.popleft()
                for dst in self.nodes:
                    extension = self.extend(index, src, dst)
                    if extension.amount:
                        frontier.append(dst)
                        if extension.remainder_id is not None:
                            queue.append(extension.remainder_id)
            batch = self._batches[index]
            if batch.vertices != frozenset(self.nodes) or len(batch.edges) != len(self.nodes) - 1:
                raise ConstructionInfeasibleError('residual cut construction did not complete a spanning tree')
            self._active.remove(index)
        items = [item for batch in self._batches.values() for item in batch.items]
        if frozenset(items) != self.original_items or len(items) != len(self.original_items):
            raise SemanticError('batch splitting lost or duplicated an integer item')
        return tuple(self._batches.values())
