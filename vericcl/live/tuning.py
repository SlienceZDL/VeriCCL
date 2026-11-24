"""Background CPU repair using the primary checker, solver and budget ledger."""
from contextlib import contextmanager
from dataclasses import dataclass, replace
from fractions import Fraction
import uuid

from vericcl.errors import SemanticError, SolverUnavailableError
from vericcl.execution.budget import BudgetExhausted
from vericcl.execution.context import DIRECTORIES, RunContext
from vericcl.execution.events import emit_event
from vericcl.input.loader import load_request
from vericcl.live.model import update_live_model
from vericcl.solver.worker_scope import isolated_solver_scope
from vericcl.topology import load_topology
from vericcl.tuning.strategies import RepairContext, RepairDomain, _network, _substitute, outside_invariants
from vericcl.tuning.resynthesis import ResynthesisRequest, resynthesize_branch
from vericcl.verification.bottlenecks import StrategyRecord, _paths
from vericcl.verification.online.model_publication import model_json
from vericcl.workflow_state import plain
from vericcl.workflow_v5 import PrimaryRequest, PrimaryWorkflow, accept_candidate
from vericcl.xml.lower import LoweringContext, lower_and_check
from vericcl.xml.v5_lower import evaluate_program


def used_links(bundle):
    return {(op.src_rank, op.dst_rank) for op in _network(bundle)}


def restricted_inputs(inputs, invalid_links, context, *, prefix='restricted'):
    config = plain(inputs.request_config); topology = config['topology']
    invalid = set(invalid_links)
    topology['directed_links'] = [edge for edge in topology['directed_links'] if (edge['src'], edge['dst']) not in invalid]
    resources = []
    for resource in topology['shared_resources']:
        resource['links'] = [edge for edge in resource['links'] if tuple(edge) not in invalid]
        if resource['links']:
            resources.append(resource)
    topology['shared_resources'] = resources
    retained = {resource['id'] for resource in resources}
    for edge in topology['directed_links']:
        edge['resources'] = [key for key in edge['resources'] if key in retained]
    files = {name: context.write_json('input_snapshot/' + prefix + '-' + name + '.json', value)
             for name, value in config.items()}
    return load_request(files, purpose='solve')


def repair_failed_links(bundle, invalid_links, lowering):
    """First try only affected transfers, preserving all outside decisions."""
    affected = [op for op in _network(bundle) if (op.src_rank, op.dst_rank) in invalid_links]
    if not affected:
        return None
    domain = RepairDomain(frozenset(op.operation_id for op in affected), 'path')
    context = RepairContext(lowering)

    def choices(index, paths):
        lowering.lease.check()
        if index == len(affected):
            strategy = StrategyRecord('path', tuple(sorted(domain.operation_ids)), paths=tuple(paths))
            candidate = lower_and_check(_substitute(bundle, strategy, context), lowering)
            if candidate.can_execute and outside_invariants(bundle, candidate, domain).valid:
                yield candidate
            return
        op = affected[index]
        for path in _paths(op.src_rank, op.dst_rank, op.participants, lowering.topology, lowering.lease):
            yield from choices(index + 1, paths + [(op.operation_id, path)])

    stream = choices(0, [])
    try:
        return next(stream, None)
    finally:
        stream.close()


@dataclass(frozen=True)
class LiveProposal:
    bundle: object
    inputs: object
    model: object
    status: str
    invalid_links: tuple
    evidence_root: str


class _BackgroundWorkflow(PrimaryWorkflow):
    def __init__(self, *args, parent_lease, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent_lease = parent_lease

    @contextmanager
    def stage(self, name):
        self.parent_lease.check()
        lease = self.budgets.acquire(name)
        lease.deadline = min(lease.deadline, self.parent_lease.deadline)
        lease.work_deadline = min(lease.work_deadline, self.parent_lease.work_deadline)
        lease.parent = self.parent_lease
        try:
            lease.check()
            with isolated_solver_scope(self.context, cleanup_deadline=lease.deadline):
                yield lease
        finally:
            lease.close(cleanup_confirmed=not self.context.active_jobs)

    def compare(self, bundle, inputs, record):
        # These are CPU proposals. The foreground controller performs any
        # requested GPU acceptance at a request boundary before committing.
        decision = accept_candidate(self.best, bundle, {})
        emit_event(self.context, 'background_candidate_checked', plain(decision))
        if decision.accepted:
            self.best, self.best_inputs, self.best_record = bundle, inputs, record
            self.processed.clear(); self.expanded.clear(); self.analysis = None
        return decision.accepted


class LiveRepairEngine:
    def __init__(self, context, budgets):
        self.context, self.budgets = context, budgets

    def __call__(self, snapshot, events, lease):
        name = uuid.uuid4().hex
        root = self.context.artifact('live/background/' + name); root.mkdir(parents=True)
        context = RunContext(root, task_id=self.context.task_id, run_id=name, redactor=self.context.redactor)
        for directory in DIRECTORIES:
            (root / directory).mkdir()
        context.write_json('run.json', {'kind': 'live_background_cpu_repair', 'base_version': snapshot['version'],
            'events': plain(events), 'parent_deadline': lease.deadline})
        emit_event(context, 'run_started')
        invalid = set(snapshot.get('invalid_links', ())) | {edge for event in events for edge in event.invalid_links}
        model = snapshot['model'] if snapshot.get('model_updated') else update_live_model(snapshot['model'], events,
            payload_bytes=snapshot['inputs'].hyperparameters.slice_size_bytes)
        inputs = restricted_inputs(snapshot['inputs'], invalid, context)
        original = restricted_inputs(snapshot.get('original', snapshot['inputs']), invalid, context, prefix='original')
        workflow = _BackgroundWorkflow(PrimaryRequest(original, cost_model=model), context,
            budgets=self.budgets, parent_lease=lease)
        context.write_json('models/' + model.version + '.json', model_json(model))
        initial = snapshot['bundle']; proposal = None; status = 'not_started'
        workflow.best_inputs = inputs
        try:
            with workflow.stage('offline') as phase:
                lowering = LoweringContext(inputs, load_topology(inputs), model, phase, run_context=context)
                workflow.best = evaluate_program(initial.source, initial.storage_plan, initial.program, initial.xml_text, lowering)
                workflow.best_record = workflow.store(workflow.best, inputs, 'live_model_reassessment')
            if used_links(initial) & invalid:
                self.budgets.start_tuning_round()
                with workflow.stage('repair') as phase:
                    lowering = replace(lowering, lease=phase)
                    proposal = repair_failed_links(initial, invalid, lowering)
                    status = 'fault_local_repair' if proposal is not None else 'no_legal_replacement'
                    if proposal is None:
                        repair = RepairContext(lowering, request_inputs=original)
                        stream = resynthesize_branch(initial, ResynthesisRequest('structure', 'failed path has no qualified local repair'), repair)
                        try:
                            for item in stream:
                                if item.candidate.can_execute and not used_links(item.candidate) & invalid:
                                    proposal, inputs, status = item.candidate, item.context.inputs, 'fault_resynthesis'
                                    break
                        finally:
                            stream.close()
                    if proposal is not None:
                        workflow.store(proposal, inputs, status)
            elif workflow.best.can_execute:
                while self.budgets.started_rounds < self.budgets.max_tuning_rounds:
                    lease.check(); self.budgets.start_tuning_round()
                    stop = workflow.tuning_round()
                    if workflow.best.xml_sha256 != initial.xml_sha256:
                        proposal, inputs, status = workflow.best, workflow.best_inputs, 'performance_proposal'
                        break
                    if stop is not None:
                        status = stop; break
                else:
                    status = 'maximum_rounds'
            else:
                status = 'current_model_unavailable'
        except BudgetExhausted as error:
            status = 'maximum_rounds' if self.budgets.started_rounds >= self.budgets.max_tuning_rounds else 'budget_exhausted'
            emit_event(context, 'background_budget_exhausted', {'message': str(error)})
        except SemanticError as error:
            if isinstance(error, SolverUnavailableError):
                raise
            status = 'no_qualified_candidate'
            emit_event(context, 'background_analysis_failed', {'message': str(error)})
        result = LiveProposal(proposal, inputs, model, status, tuple(sorted(invalid)), str(root))
        context.write_json('reports/proposal.json', {'status': status, 'base_version': snapshot['version'],
            'candidate_version': None if proposal is None else proposal.xml_sha256, 'model_version': model.version,
            'invalid_links': plain(invalid), 'gpu_acceptance': 'pending' if proposal is not None else 'not_applicable',
            'budgets': self.budgets.snapshot(), 'active_jobs': sorted(context.active_jobs)})
        emit_event(context, 'background_finished', {'status': status})
        return result
