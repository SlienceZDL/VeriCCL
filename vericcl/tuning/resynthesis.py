"""Structural and partition branches share the existing repair lease and incumbent."""
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
        if self.kind not in {'structure', 'partition'} or not isinstance(self.reason, str) or not self.reason:
            raise SemanticError('resynthesis requires a structural/partition branch and its benefit rationale')
        if (self.kind == 'partition') != (self.slice_bytes is not None):
            raise SemanticError('only a partition branch supplies a new slice size')


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


def resynthesize_branch(incumbent, request, context):
    return ResynthesisStream(incumbent, request, context)
