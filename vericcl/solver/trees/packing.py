"""Exact bandwidth packing and lazy integer route-column optimization."""
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import gcd, lcm

from vericcl.errors import ConstructionInfeasibleError, SemanticError
from .batch import BatchPacker, rooted_cut_feasible, _check


@dataclass(frozen=True)
class BandwidthPacking:
    batches: tuple
    bandwidth_time_us: Fraction
    exact_optimal: bool = True
    scope: str = 'integer_rooted_trees_without_shared_resources'


def pack_bandwidth(nodes, edge_slice_times, root_items, lease=None):
    """Minimize max_e(copies_e * per_slice_time_e), ignoring startup.

    Every breakpoint is an integer multiple of a rational common time unit.
    Binary search over those integer units and exact integer min-cuts avoids
    treating a floating solver tolerance as a routing proof.
    """
    nodes = tuple(nodes)
    times = {edge: Fraction(value) for edge, value in edge_slice_times.items()}
    if any(time <= 0 for time in times.values()):
        raise SemanticError('bandwidth edge slice times must be positive')
    counts = {root: len(items) for root, items in root_items.items() if items}
    total = sum(counts.values())
    if len(nodes) == 1:
        return BandwidthPacking(BatchPacker(nodes, {}, root_items, lease).finish(), Fraction(0))
    if not times or not rooted_cut_feasible(nodes, {edge: total for edge in times}, counts, lease):
        raise ConstructionInfeasibleError('rooted cut is unreachable in the directed graph')
    denominator = lcm(*(time.denominator for time in times.values()))
    numerators = [time.numerator * (denominator // time.denominator) for time in times.values()]
    unit = Fraction(reduce(gcd, numerators), denominator)
    steps = {edge: int(time / unit) for edge, time in times.items()}
    def capacities(tick):
        return {edge: min(total, tick // period) for edge, period in steps.items()}
    low, high = -1, max(steps.values()) * total
    while high - low > 1:
        _check(lease)
        middle = (low + high) // 2
        if rooted_cut_feasible(nodes, capacities(middle), counts, lease):
            high = middle
        else:
            low = middle
    batches = BatchPacker(nodes, capacities(high), root_items, lease).finish()
    return BandwidthPacking(batches, high * unit)
