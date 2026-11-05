"""Lazy hierarchical plan search, equivalent demands, and representative solves.

These candidates precede buffer/TB lowering. Their communication estimate is
kept distinct from the complete executable T_pred produced by that later gate.
"""
from dataclasses import dataclass, field, replace
from itertools import product, chain, islice
from collections import defaultdict
from contextlib import contextmanager
import time

from vericcl.errors import ConstructionInfeasibleError, InputValidationError, SemanticError
from vericcl.execution.budget import BudgetExhausted, StandaloneLease
from vericcl.input.models import ObjectiveMode
from vericcl.input.json_codec import sha256_json
from vericcl.planner.model import PlanDAG, PlanEdge, PlanNode, PlanningMode, StageInterface
from vericcl.planner.direct import build_direct_plan, _initial_inputs
from vericcl.planner.manual import manual_enabled, plan_manual
from vericcl.semantics.collective import CollectiveKind, CollectiveSpec, OutputSlot, required_outputs


@dataclass(frozen=True)
class HierarchyOptions:
    symmetry: bool
    shortest_paths: bool
    batching: bool
    batch_slices: int = 1
    reuse_subproblems: bool = True


def hierarchy_options(inputs):
    strategy = inputs.strategies
    return HierarchyOptions(strategy.symmetry, strategy.shortest_paths, strategy.batching,
                            inputs.hyperparameters.slice_count if strategy.batching else 1)


@dataclass(frozen=True)
class HierarchyPlan:
    plan: PlanDAG
    rails: tuple = ()
    rail_assignments: tuple = ()
    root_assignments: tuple = ()
    search_scope: str = 'topology_group_hierarchy'


@dataclass
class HierarchyContext:
    lease: object
    reuse_subproblems: bool = True
    # A bounded coarse batch limits working memory, not the candidate space.
    coarse_batch_size: int = 4
    records: list = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.coarse_batch_size, bool) or not isinstance(self.coarse_batch_size, int) or self.coarse_batch_size < 1:
            raise SemanticError('coarse batch size must be positive')


@dataclass(frozen=True)
class HierarchyCandidate:
    plan_spec: HierarchyPlan
    schedule: object
    templates: tuple
    patterns: object
    communication_time_us: float
    model_version: str
    stage: str
    full_search_proven: bool = False
    runtime_checked: bool = False

    @property
    def plan(self):
        return self.plan_spec.plan


def _check(lease):
    if lease is not None:
        lease.check()


def discover_rails(topology, lease=None):
    groups = defaultdict(list)
    for rank, node in topology.node_membership.items():
        groups[node].append(rank)
    groups = tuple(tuple(sorted(ranks)) for _, ranks in sorted(groups.items()))
    if len(groups) <= 1:
        return ()
    gateways = {key.src_rank for key in topology.links
                if topology.node_membership[key.src_rank] != topology.node_membership[key.dst_rank]}
    gateways |= {key.dst_rank for key in topology.links
                 if topology.node_membership[key.src_rank] != topology.node_membership[key.dst_rank]}
    choices = [tuple(rank for rank in group if rank in gateways) for group in groups]
    rails = []
    for selection in product(*choices):
        _check(lease)
        selected = set(selection)
        reached = {selection[0]}
        while True:
            expanded = reached | {key.dst_rank for key in topology.links if key.src_rank in reached and key.dst_rank in selected}
            expanded |= {key.src_rank for key in topology.links if key.dst_rank in reached and key.src_rank in selected}
            if reached == expanded:
                break
            reached = expanded
        if reached == selected:
            rails.append(tuple(sorted(selection)))
    return tuple(sorted(set(rails)))


def integer_allocations(total, choices):
    if isinstance(total, bool) or not isinstance(total, int) or total < 0 or choices < 1:
        raise SemanticError('integer allocation needs nonnegative slices and positive choices')
    if choices == 1:
        yield (total,)
    else:
        for amount in range(total + 1):
            for suffix in integer_allocations(total - amount, choices - 1):
                yield (amount,) + suffix


def _tasks(inputs):
    outputs = required_outputs(inputs.collective, inputs.rank_count, inputs.hyperparameters.slice_count)
    by_contributors = defaultdict(dict)
    for slot, contributors in outputs.items():
        by_contributors[contributors][slot] = contributors
    return tuple((contributors, StageInterface(values)) for contributors, values in
                 sorted(by_contributors.items(), key=lambda item: tuple(sorted(item[0]))))


def _node(inputs, topology, node_id, stage, kind, group, source, destination, root):
    group = tuple(sorted(group))
    links = frozenset(key for key in topology.links if key.src_rank in group and key.dst_rank in group)
    contributors = set().union(*source.values.values(), *destination.values.values())
    spec = CollectiveSpec(kind, inputs.collective.datatype,
        inputs.collective.reduction_op if kind is CollectiveKind.REDUCE else None, root, inputs.collective.inplace)
    return PlanNode(node_id, stage, spec, group, source, destination, links,
                    frozenset(resource for key in links for resource in topology.resources_for(key)),
                    participants_by_slice={contributor: group for contributor in contributors})


def build_group_plan(inputs, topology, rails, assignments, roots=None):
    """Gather/reduce locally, communicate between gateways, then distribute.

    Each complete slice chooses one rail. A rail may reuse a gateway that also
    belongs to another rail; global resource composition accounts for that.
    """
    tasks = _tasks(inputs)
    if len(assignments) != len(tasks) or any(not 0 <= a < len(rails) for a in assignments):
        raise SemanticError('rail assignments must cover every logical demand')
    groups = defaultdict(list)
    for rank, node in topology.node_membership.items():
        groups[node].append(rank)
    group_order = tuple(sorted(groups))
    n = inputs.hyperparameters.slice_count
    nodes, edges, actual_roots = [], [], []
    allreduce = inputs.collective.kind is CollectiveKind.ALL_REDUCE
    for task_index, ((contributors, final), rail_id) in enumerate(zip(tasks, assignments)):
        rail = rails[rail_id]
        gateways = {topology.node_membership[rank]: rank for rank in rail}
        if set(gateways) != set(groups):
            raise SemanticError('rail must explicitly cover every topology group')
        prefix = 'chunk-%08d' % task_index
        temporary = min(contributors) if len(contributors) == 1 else min(contributors) % n
        local_nodes, inter_values = [], {}
        reduction = len(contributors) > 1
        for group_id in group_order:
            original = {OutputSlot(slice_id // n, slice_id % n): frozenset({slice_id})
                        for slice_id in contributors if slice_id // n in groups[group_id]}
            if not original:
                continue
            gateway = gateways[group_id]
            subset = frozenset().union(*original.values())
            output = StageInterface({OutputSlot(gateway, temporary): subset})
            node = _node(inputs, topology, prefix + '-local-%d' % group_id, 0,
                         CollectiveKind.REDUCE if reduction else CollectiveKind.GATHER,
                         groups[group_id], StageInterface(original), output, gateway)
            nodes.append(node)
            local_nodes.append(node)
            inter_values.update(output.values)
        targets = {topology.node_membership[slot.rank] for slot in final.values}
        if allreduce:
            chosen_root = rail[task_index % len(rail)] if roots is None else roots[task_index]
        elif reduction:
            chosen_root = gateways[next(iter(targets))]
        else:
            chosen_root = next(iter(inter_values)).rank
        if chosen_root not in rail:
            raise SemanticError('internal root must belong to the selected rail')
        actual_roots.append(chosen_root)
        inter_output = StageInterface({OutputSlot(chosen_root, temporary): contributors} if reduction else
            {OutputSlot(gateways[group], temporary): contributors for group in targets})
        inter = _node(inputs, topology, prefix + '-inter', 1,
                      CollectiveKind.REDUCE if reduction else CollectiveKind.BROADCAST,
                      rail, StageInterface(inter_values), inter_output, chosen_root)
        nodes.append(inter)
        edges.extend(PlanEdge(node.node_id, inter.node_id, node.logical_output) for node in local_nodes)
        distributor = inter
        if allreduce:
            inter_output = StageInterface({OutputSlot(gateway, temporary): contributors for gateway in rail})
            distributor = _node(inputs, topology, prefix + '-inter-distribute', 2, CollectiveKind.BROADCAST,
                                rail, inter.logical_output, inter_output, chosen_root)
            nodes.append(distributor)
            edges.append(PlanEdge(inter.node_id, distributor.node_id, inter.logical_output))
        for group_id in sorted(targets):
            gateway = gateways[group_id]
            source = StageInterface({OutputSlot(gateway, temporary): contributors})
            final_values = {slot: value for slot, value in final.values.items() if slot.rank in groups[group_id]}
            last = _node(inputs, topology, prefix + '-output-%d' % group_id, 3 if allreduce else 2,
                         CollectiveKind.BROADCAST, groups[group_id], source, StageInterface(final_values), gateway)
            nodes.append(last)
            edges.append(PlanEdge(distributor.node_id, last.node_id, source))
    plan = PlanDAG(inputs.collective, inputs.rank_count, n, _initial_inputs(inputs.rank_count, n),
                   tuple(nodes), tuple(edges), StageInterface(required_outputs(inputs.collective, inputs.rank_count, n)),
                   PlanningMode.HIERARCHY, 'topology_group_search')
    return HierarchyPlan(plan, tuple(rails), tuple(assignments), tuple(actual_roots))


def _explicit_roles(plan):
    nodes = tuple(replace(node, participants_by_slice={slice_id: node.communication_group
        for interface in (node.logical_input, node.logical_output) for contributors in interface.values.values()
        for slice_id in contributors}) for node in plan.nodes)
    return replace(plan, nodes=nodes)


def _valid_stages(plan, requested):
    stages = {node.stage_id for node in plan.nodes}
    return stages == set(range(len(stages))) and (requested is None or len(stages) == requested)


def plan_identity(plan):
    def interface(value):
        return tuple((slot.rank, slot.offset, tuple(sorted(contributors)))
                     for slot, contributors in sorted(value.values.items()))
    return sha256_json({'collective': plan.collective, 'ranks': plan.rank_count, 'N': plan.slice_count,
        'nodes': [(node.node_id, node.stage_id, node.local_collective, node.communication_group,
                   interface(node.logical_input), interface(node.logical_output), sorted(node.allowed_links),
                   sorted(node.participants_by_slice.items())) for node in plan.nodes],
        'edges': [(edge.producer_id, edge.consumer_id, interface(edge.interface)) for edge in plan.edges]})


def _enumerate_partition_plans(inputs, topology, *, lease=None):
    if inputs.hyperparameters.slice_count is None:
        raise InputValidationError('choose S/C before hierarchical synthesis')
    if manual_enabled(inputs):
        yield HierarchyPlan(plan_manual(inputs, topology), search_scope='manual_plan')
        return
    requested = inputs.atom_constraints.stage_num
    rails = discover_rails(topology, lease)
    count = len(_tasks(inputs))
    seen = set()
    def build(assignment, roots=None):
        _check(lease)
        spec = build_group_plan(inputs, topology, rails, assignment, roots)
        if not _valid_stages(spec.plan, requested):
            return None
        key = assignment, spec.root_assignments
        if key in seen:
            return None
        seen.add(key)
        return spec
    if rails:
        balanced = tuple(index % len(rails) for index in range(count))
        spec = build(balanced)
        if spec is not None:
            yield spec
    direct_inputs = replace(inputs, strategies=replace(inputs.strategies, hierarchy=False, manual_hierarchy=()))
    direct = _explicit_roles(build_direct_plan(direct_inputs, topology, 'direct_search_candidate'))
    if _valid_stages(direct, requested):
        yield HierarchyPlan(direct)
    if inputs.collective.kind is CollectiveKind.ALL_REDUCE and requested == 1:
        # One logical AllReduce stage still has reduction then distribution
        # dependencies; stage labels never erase those data dependencies.
        yield HierarchyPlan(replace(direct, nodes=tuple(replace(node, stage_id=0) for node in direct.nodes)))
    if not rails:
        return
    if requested is not None and requested != (4 if inputs.collective.kind is CollectiveKind.ALL_REDUCE else 3):
        return
    seeds = [tuple([rail] * count) for rail in range(len(rails))]
    for assignment in chain(seeds, product(range(len(rails)), repeat=count)):
        spec = build(tuple(assignment))
        if spec is not None:
            yield spec
        if inputs.collective.kind is CollectiveKind.ALL_REDUCE:
            for roots in product(*(rails[rail] for rail in assignment)):
                spec = build(tuple(assignment), roots)
                if spec is not None:
                    yield spec


def candidate_partitions(inputs, topology, *, lease=None):
    """Topology node groups, explicit group covers, and capacity-layer components.

    This is an initial search grammar, not a proof over all rank partitions.
    """
    partitions = []
    seen = set()
    def add(groups):
        value = tuple(sorted(tuple(sorted(group)) for group in groups))
        if value not in seen:
            seen.add(value)
            partitions.append(value)
    by_node = defaultdict(list)
    for rank, node in topology.node_membership.items():
        by_node[node].append(rank)
    add(by_node.values())
    explicit = tuple(tuple(group) for group in inputs.resolved_topology.get('groups', ()))
    def covers(remaining, groups):
        _check(lease)
        if not remaining:
            add(groups)
            return
        first = min(remaining)
        for group in explicit:
            if first in group and set(group) <= remaining:
                covers(remaining - set(group), groups + [group])
    if explicit:
        covers(set(range(topology.rank_count)), [])
    from vericcl.solver.scheduling import curve_duration_us
    costs = {key: curve_duration_us(edge.performance, inputs.hyperparameters.slice_size_bytes, 1)
             for key, edge in topology.links.items()}
    for threshold in sorted(set(costs.values())):
        _check(lease)
        remaining = set(range(topology.rank_count))
        groups = []
        while remaining:
            component = {min(remaining)}
            while True:
                grown = component | {key.dst_rank for key, cost in costs.items() if cost <= threshold and key.src_rank in component}
                grown |= {key.src_rank for key, cost in costs.items() if cost <= threshold and key.dst_rank in component}
                if grown == component:
                    break
                component = grown
            remaining -= component
            groups.append(component)
        add(groups)
    return tuple(partitions)


def coalesce_plan_batches(plan):
    """Group compatible logical tasks; each wire step still carries one slice."""
    classes = []
    for node in plan.nodes:
        key = (node.stage_id, node.local_collective, node.communication_group, node.allowed_links,
               tuple(sorted(node.shared_resource_ids)), bool(node.dual_of_node_id))
        for saved in classes:
            if key != saved['key']:
                continue
            if any(slot in saved[name] and saved[name][slot] != value
                   for name, interface in (('inputs', node.logical_input), ('outputs', node.logical_output))
                   for slot, value in interface.values.items()):
                continue
            if any(slice_id in saved['roles'] and saved['roles'][slice_id] != ranks
                   for slice_id, ranks in node.participants_by_slice.items()):
                continue
            break
        else:
            saved = {'key': key, 'members': [], 'inputs': {}, 'outputs': {}, 'roles': {}}
            classes.append(saved)
        saved['members'].append(node)
        saved['inputs'].update(node.logical_input.values)
        saved['outputs'].update(node.logical_output.values)
        saved['roles'].update(node.participants_by_slice)
    nodes, mapping = [], {}
    for index, saved in enumerate(classes):
        node_id = 'batch-node-%08d' % index
        for node in saved['members']:
            mapping[node.node_id] = node_id
        nodes.append(replace(saved['members'][0], node_id=node_id,
            logical_input=StageInterface(saved['inputs']), logical_output=StageInterface(saved['outputs']),
            participants_by_slice=saved['roles']))
    nodes = tuple(replace(node, dual_of_node_id=mapping.get(node.dual_of_node_id, node.dual_of_node_id)) for node in nodes)
    edges = defaultdict(dict)
    for edge in plan.edges:
        key = mapping[edge.producer_id], mapping[edge.consumer_id]
        if key[0] == key[1]:
            raise SemanticError('batching cannot collapse a data dependency')
        edges[key].update(edge.interface.values)
    return replace(plan, nodes=nodes, edges=tuple(PlanEdge(src, dst, StageInterface(values)) for (src, dst), values in sorted(edges.items())))


def enumerate_hierarchy_plans(inputs, topology, *, lease=None):
    if manual_enabled(inputs):
        yield HierarchyPlan(plan_manual(inputs, topology), search_scope='manual_plan')
        return
    if inputs.hyperparameters.slice_size_bytes is None:
        raise InputValidationError('choose S/C before hierarchical synthesis')
    iterators = []
    for partition in candidate_partitions(inputs, topology, lease=lease):
        membership = {rank: index for index, group in enumerate(partition) for rank in group}
        projected = replace(topology, node_membership=membership, isomorphism_signature='')
        iterators.append(iter(_enumerate_partition_plans(inputs, projected, lease=lease)))
    seen = set()
    while iterators:
        pending = []
        for iterator in iterators:
            _check(lease)
            candidate = next(iterator, None)
            if candidate is None:
                continue
            pending.append(iterator)
            if inputs.strategies.batching:
                candidate = replace(candidate, plan=coalesce_plan_batches(candidate.plan))
            identity = plan_identity(candidate.plan)
            if identity not in seen:
                seen.add(identity)
                yield candidate
        iterators = pending


class CandidateStream:
    def __init__(self, inputs, topology, model, context):
        self.inputs, self.topology, self.model, self.context = inputs, topology, model, context
        self.model_calls = 0
        self.fine_model_calls = 0
        self.template_hits = 0
        self.stop_reason = None
        self._used = False

    def __iter__(self):
        if self._used:
            raise SemanticError('candidate stream is single-use')
        self._used = True
        try:
            yield from self._generate()
            self.stop_reason = 'initial_grammar_exhausted'
        except BudgetExhausted:
            self.stop_reason = 'initial_synthesis_budget'

    @contextmanager
    def _model_budget(self, inputs, *, fine):
        from vericcl.solver.budget import ModelBudget
        parent = self.context.lease
        parent.check()
        child = None
        if hasattr(parent, 'manager'):
            child = parent.manager.acquire(parent.stage, model=True, parent=parent)
            lease = child
            threads = child.threads
        else:
            lease = parent
            from vericcl.execution.budget import effective_cpu_budget
            threads = min(inputs.solver.max_threads_per_model, effective_cpu_budget(inputs))
        limited = None
        try:
            now = time.monotonic()
            seconds = min(inputs.solver.per_model_timeout_s * (1 if fine else .25), lease.remaining_seconds())
            class LimitedLease(StandaloneLease):
                @property
                def cancelled(self):
                    return lease.cancelled
            limited = LimitedLease(now + seconds)
            limited.hard_deadline = lease.deadline
            configured = replace(inputs, solver=replace(inputs.solver, max_threads_per_model=threads))
            yield configured, ModelBudget(seconds, now, now + seconds, limited)
        finally:
            if child is not None:
                child.close(cleanup_confirmed=not getattr(limited, 'ownership_unresolved', False))

    def _solve(self, spec, topology, *, fine):
        from vericcl.solver.demands import build_solver_problem
        from vericcl.solver.templates import build_solver_templates
        from vericcl.solver.routing_milp import solve_route_milp
        from vericcl.solver.hierarchy_fine import solve_fine_pattern
        from vericcl.solver.instantiate import instantiate_route_patterns
        from vericcl.composer import compose_routes
        inputs = self.inputs
        channels = inputs.solver.max_channels
        self.context.lease.check()
        problems = tuple(build_solver_problem(node, inputs, topology) for node in spec.plan.nodes)
        if any(problem.infeasible_demand_ids for problem in problems):
            self.context.records.append({'plan': plan_identity(spec.plan), 'status': 'unreachable_demand'})
            return None
        templates = build_solver_templates(problems, spec.plan.planning_mode, reuse=self.context.reuse_subproblems)
        by_node = {problem.node.node_id: problem for problem in problems}
        patterns = {}
        batch_results = {}
        objective = inputs.hyperparameters.objective_mode
        if objective is ObjectiveMode.AUTO:
            objective = ObjectiveMode.LATENCY
        try:
            for template in templates:
                node_id = template.representative.node.node_id
                if fine and inputs.strategies.batching and node_id in batch_results:
                    patterns[template.template_id] = solve_fine_pattern(template, by_node[node_id], channels,
                        objective, None, result=batch_results[node_id])
                    continue
                with self._model_budget(inputs, fine=fine) as (configured, budget):
                    self.model_calls += 1
                    if fine:
                        self.fine_model_calls += 1
                        problem = replace(by_node[node_id], inputs=configured)
                        if inputs.strategies.batching:
                            from vericcl.solver.milp import solve_milp
                            batch_results[node_id] = solve_milp(problem, channels, objective, budget, None)
                        pattern = solve_fine_pattern(template, problem, channels, objective, budget,
                                                     result=batch_results.get(node_id))
                    else:
                        pattern = solve_route_milp(template, configured, topology, channels, objective, budget)
                    patterns[template.template_id] = pattern
            self.context.lease.check()
            instantiated = instantiate_route_patterns(spec.plan, templates, patterns, inputs, topology)
            if instantiated.failures:
                raise ConstructionInfeasibleError(str(instantiated.failures))
            schedule = compose_routes(spec.plan, instantiated.node_schedules, topology, channels,
                                      hierarchy_event_policy=True, lease=self.context.lease)
        except ConstructionInfeasibleError as error:
            self.context.records.append({'plan': plan_identity(spec.plan), 'phase': 'fine' if fine else 'coarse',
                                         'status': 'no_incumbent', 'detail': str(error)})
            return None
        self.template_hits += sum(len(template.members) - 1 for template in templates)
        estimate = max((transfer.ed_time for transfer in schedule.transfers), default=0.)
        phase = 'fine' if fine else 'coarse'
        self.context.records.append({'plan': plan_identity(spec.plan), 'status': phase + '_candidate',
                                     'model_calls': self.model_calls, 'communication_time_us': estimate})
        return HierarchyCandidate(spec, schedule, templates, patterns, estimate,
            'initial_topology' if self.model is None else self.model.version, phase)

    def _generate(self):
        from vericcl.solver.scip_api import ScipAdapter
        ScipAdapter.require()
        topology = self.topology
        if self.model is not None:
            from vericcl.performance.topology import project_topology
            topology = project_topology(topology, self.model, self.inputs.hyperparameters.slice_size_bytes)
        plans = iter(enumerate_hierarchy_plans(self.inputs, topology, lease=self.context.lease))
        while True:
            batch = tuple(islice(plans, self.context.coarse_batch_size))
            if not batch:
                return
            coarse = []
            failed = []
            for spec in batch:
                candidate = self._solve(spec, topology, fine=False)
                if candidate is not None:
                    coarse.append(candidate)
                    yield candidate
                else:
                    failed.append(spec)
            # The finite coarse batch only sets priority. It never certifies
            # that a later candidate, or a route absent from coarse paths, loses.
            ranked = [candidate.plan_spec for candidate in sorted(coarse, key=lambda item: item.communication_time_us)]
            for spec in chain(ranked, failed):
                candidate = self._solve(spec, topology, fine=True)
                if candidate is not None:
                    yield candidate


def plan_hierarchy(inputs, topology, model, context):
    if not isinstance(context, HierarchyContext):
        raise SemanticError('hierarchy needs an explicit context and parent budget')
    return CandidateStream(inputs, topology, model, context)
