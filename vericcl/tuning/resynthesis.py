"""Structural, partition and refinement branches share the existing repair lease and incumbent.

A refinement branch keeps the incumbent's routes, channels and TB order and
only cuts every slice into ``factor`` consecutive pieces (S -> S/factor). The
runtime pipelines consecutive steps of a TB through its connection FIFO, so
finer pieces can overlap transfers along a route that whole slices serialize;
each extra step also pays the per-step overhead. Whether that helps is decided
like any other S change: the refined candidate is lowered and checked through
the normal path and must improve under the model and the paired measurement.
"""
from dataclasses import dataclass, replace

from vericcl.errors import SemanticError
from vericcl.planner.parameters import initial_parameter_domain, resolve_parameters
from vericcl.planner.group_search import HierarchyContext, plan_hierarchy
from vericcl.solver.constructive import construct_routes, optimize_fixed_routes
from vericcl.solver.trees.operators import RouteContext
from vericcl.topology import load_topology
from vericcl.xml.lower import lower_and_check


@dataclass(frozen=True)
class ResynthesisRequest:
    kind: str
    reason: str
    slice_bytes: int | None = None
    channels: int | None = None

    def __post_init__(self):
        if self.kind not in {'structure', 'partition', 'refinement'} or not isinstance(self.reason, str) or not self.reason:
            raise SemanticError('resynthesis requires a structural/partition/refinement branch and its benefit rationale')
        if (self.kind in {'partition', 'refinement'}) != (self.slice_bytes is not None):
            raise SemanticError('only a partition or refinement branch supplies a new slice size')


@dataclass(frozen=True)
class ResynthesisCandidate:
    incumbent: object
    candidate: object
    context: object
    request: ResynthesisRequest
    accepted: bool = False


class ResynthesisStream:
    budget_stage = 'repair'
    outer_round_reset = False

    def __init__(self, incumbent, request, context):
        self.incumbent, self.request, self.context = incumbent, request, context
        self._iterator = self._run()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    def close(self):
        self._iterator.close()

    def _run(self):
        parent = self.context.lowering
        parent.lease.check()
        original = self.context.request_inputs or parent.inputs
        if self.request.kind == 'refinement':
            yield from self._refine(parent, original)
            return
        size = self.request.slice_bytes if self.request.kind == 'partition' else parent.inputs.hyperparameters.slice_size_bytes
        if self.request.kind == 'partition' and size == parent.inputs.hyperparameters.slice_size_bytes:
            raise SemanticError('partition branch must use a different slice size')
        domain = initial_parameter_domain(original, fixed_slice_bytes=size, lease=parent.lease)
        selected = [candidate for candidate in domain.candidates if self.request.channels is None or candidate.channels == self.request.channels]
        if not selected:
            raise SemanticError('requested resynthesis parameters violate a necessary runtime or partition constraint')
        for choice in selected:
            parent.lease.check()
            inputs = resolve_parameters(original, choice)
            topology = load_topology(inputs)
            lowering = replace(parent, inputs=inputs, topology=topology)
            self.context.records.append({'phase': 'repair_resynthesis', 'kind': self.request.kind,
                'slice_bytes': choice.slice_bytes, 'channels': choice.channels, 'reason': self.request.reason,
                'budget_stage': 'repair', 'outer_round_reset': False, 'incumbent_retained': True})
            if inputs.selected_synthesis_method == 'hierarchy':
                child = HierarchyContext(parent.lease, records=self.context.records)
                stream = iter(plan_hierarchy(inputs, topology, parent.model, child))
            else:
                child = RouteContext(parent.lease, records=self.context.records)
                stream = iter(optimize_fixed_routes(construct_routes(inputs, topology, parent.model, child), child))
            try:
                for candidate in stream:
                    parent.lease.check()
                    result = lower_and_check(candidate, lowering)
                    yield ResynthesisCandidate(self.incumbent, result, lowering, self.request)
            finally:
                if hasattr(stream, 'close'):
                    stream.close()


    def _refine(self, parent, original):
        from vericcl.tuning.correctness import network_operations
        from vericcl.xml.v5_model import NetworkCandidate
        old = parent.inputs.hyperparameters
        size = self.request.slice_bytes
        if not old.slice_size_bytes or size >= old.slice_size_bytes or old.slice_size_bytes % size:
            raise SemanticError('a refinement must divide the incumbent slice size')
        factor = old.slice_size_bytes // size
        operations, _ = network_operations(self.incumbent, parent)
        needed = max((op.channel for op in operations), default=0) + 1
        domain = initial_parameter_domain(original, fixed_slice_bytes=size, lease=parent.lease)
        choices = sorted((c for c in domain.candidates if c.channels >= needed and
                          (self.request.channels is None or c.channels == self.request.channels)),
                         key=lambda c: (c.channels != parent.inputs.solver.max_channels, c.channels))
        if not choices:
            raise SemanticError('the refined slice size violates a necessary runtime or partition constraint')
        inputs = resolve_parameters(original, choices[0])
        lowering = replace(parent, inputs=inputs, topology=load_topology(inputs))
        self.context.records.append({'phase': 'repair_resynthesis', 'kind': 'refinement', 'slice_bytes': size,
            'factor': factor, 'channels': choices[0].channels, 'reason': self.request.reason,
            'budget_stage': 'repair', 'outer_round_reset': False, 'incumbent_retained': True})
        refined = refine_network(operations, old.slice_count, factor, inputs.hyperparameters.slice_count)
        candidate = NetworkCandidate(inputs.rank_count, inputs.hyperparameters.slice_count, size, refined, 'refinement')
        parent.lease.check()
        yield ResynthesisCandidate(self.incumbent, lower_and_check(candidate, lowering), lowering, self.request)


def refine_network(operations, old_count, factor, new_count):
    """The same transfers with every slice cut into ``factor`` consecutive pieces.

    Global slice ids are ``owner * count + position``; position ``p`` becomes
    positions ``p * factor .. p * factor + factor - 1``, so every piece stays
    inside its original byte range (hard restrictions keep binding). Piece
    ``i`` of an operation depends on piece ``i`` of its dependencies, and the
    pieces keep the operation's route, channel, stage and priority, so a TB
    runs an operation's pieces back to back where it ran the whole slice.
    """
    if factor < 2 or new_count != old_count * factor:
        raise SemanticError('refinement needs an integer factor of at least two')

    def pieces(values, index):
        return frozenset((value // old_count) * new_count + (value % old_count) * factor + index for value in values)

    def name(key, index):
        return '%s-p%02d' % (key, index)
    return tuple(replace(op, operation_id=name(op.operation_id, index),
                         contributors=pieces(op.contributors, index),
                         target_contributors=pieces(op.target_contributors, index),
                         dependencies=frozenset(name(key, index) for key in op.dependencies),
                         reference_dependencies=frozenset(name(key, index) for key in op.reference_dependencies))
                 for op in operations for index in range(factor))


def resynthesize_branch(incumbent, request, context):
    return ResynthesisStream(incumbent, request, context)
