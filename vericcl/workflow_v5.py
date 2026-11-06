"""The v5 synthesis/calibration/analysis/repair lifecycle and adoption contract."""
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, replace
from fractions import Fraction
import json
import hashlib
from pathlib import Path
import time
import uuid

from vericcl.errors import ConstructionInfeasibleError, ResourceCleanupError, SemanticError, SolverUnavailableError
from vericcl.execution.budget import BudgetExhausted, StandaloneLease
from vericcl.execution.context import StorageFailure
from vericcl.execution.events import emit_event
from vericcl.execution.status import RunResult
from vericcl.performance.initial import initial_cost_model
from vericcl.performance.serialization import load_model
from vericcl.planner.parameters import bind_artifact_partition, initial_parameter_domain, resolve_parameters
from vericcl.planner.group_search import HierarchyContext, plan_hierarchy
from vericcl.solver.constructive import construct_routes, optimize_fixed_routes
from vericcl.solver.trees.operators import RouteContext
from vericcl.solver.worker_scope import isolated_solver_scope
from vericcl.topology import load_topology
from vericcl.tuning.resynthesis import ResynthesisRequest, resynthesize_branch
from vericcl.tuning.strategies import RepairContext, build_domain, solve_repair
from vericcl.verification.bottlenecks import AnalysisContext, SearchScope, analyze
from vericcl.verification.global_bound import certify_global_cut, check_global_cut
from vericcl.verification.online.campaign import run_calibration
from vericcl.verification.online.probe_jobs import ProbeContext, ProbeFlow, _mode, payload_sizes
from vericcl.verification.online.model_publication import model_json
from vericcl.workflow_state import build_budgets, save_checkpoint, persist_candidate, checkpoint_inputs, plain
from vericcl.xml.lower import lower_and_check, LoweringContext
from vericcl.xml.parser import check_xml_bundle
from vericcl.xml.v5_lower import rejected_xml_bundle


@dataclass(frozen=True)
class PrimaryRequest:
    inputs: object
    purpose: str = 'solve'
    xml_text: str | None = None
    sidecar: dict | None = None
    tune: bool = False
    cost_model: object = None


@dataclass
class WorkflowServices:
    probe_driver: object = None
    native_driver: object = None


@dataclass(frozen=True)
class CandidateDecision:
    accepted: bool
    status: str
    evidence: dict


def accept_candidate(old, new, evidence):
    if not new.can_execute or new.prediction is None or new.prediction.exact_completion_ratio is None:
        return CandidateDecision(False, 'required_checks_failed', evidence)
    if old is not None:
        if not old.can_execute or old.prediction.model_version != new.prediction.model_version:
            return CandidateDecision(False, 'model_mismatch', evidence)
        old_contract, new_contract = old.sidecar['request_contract'], new.sidecar['request_contract']
        if any(old_contract[key] != new_contract[key] for key in old_contract if key != 'slice_bytes'):
            return CandidateDecision(False, 'external_contract_changed', evidence)
        if Fraction(*new.prediction.exact_completion_ratio) >= Fraction(*old.prediction.exact_completion_ratio):
            return CandidateDecision(False, 'no_model_gain', evidence)
    if evidence.get('measured_required'):
        if not evidence.get('trace_complete') or evidence.get('trace_evidence_kind') != 'native_msccl':
            return CandidateDecision(False, 'required_trace_incomplete', evidence)
        if old is not None and not evidence.get('measured_accepted'):
            return CandidateDecision(False, 'required_measured_gain_absent', evidence)
    return CandidateDecision(True, 'baseline' if old is None else 'improved', evidence)


def _channel_cap(inputs, channels):
    from vericcl.input.schema import _freeze
    config = plain(inputs.request_config)
    config['sketch']['solver']['max_channels'] = channels
    return replace(inputs, solver=replace(inputs.solver, max_channels=channels), request_config=_freeze(config))


class PrimaryWorkflow:
    def __init__(self, request, context, services=None, *, restored=None, budgets=None):
        self.request = request if isinstance(request, PrimaryRequest) else PrimaryRequest(request)
        self.original = self.request.inputs
        self.context = context
        self.services = services or WorkflowServices()
        self.budgets = budgets or build_budgets(self.original, purpose=self.request.purpose)
        self.initial_model = self.request.cost_model or initial_cost_model(self.original)
        self.model = self.initial_model
        self.best = self.best_inputs = self.best_record = None
        self.best_trace = None
        self.candidates = []
        self.analysis = None
        self.processed = set()
        self.expanded = set()
        self.failures = []
        self.proof = None
        self.phase = 'start'
        self.restored = restored
        self.calibrated = False
        self.calibration_status = 'not_run'
        if restored:
            self.model = load_model(context.artifact(restored['model']))
            self.best_inputs = checkpoint_inputs(context, restored)
            self.best_record = restored['bundle']
            self.best_trace = restored.get('extra', {}).get('best_trace')
            self.processed = set(restored['search'].get('processed', ()))
            self.expanded = set(restored['search'].get('expanded', ()))
            self.calibrated = restored.get('extra', {}).get('calibrated', False)
            initial_version = restored.get('extra', {}).get('initial_model_version')
            if initial_version:
                self.initial_model = load_model(context.artifact('models/' + initial_version + '.json'))

    def checkpoint(self):
        return save_checkpoint(self.context, self.best_inputs or self.original, self.budgets, self.model,
            self.best, phase=self.phase, search={'processed': sorted(self.processed), 'expanded': sorted(self.expanded)},
            bundle_record=self.best_record, extra={'purpose': self.request.purpose, 'tune': self.request.tune, 'calibrated': self.calibrated,
                                                  'original_config': plain(self.original.request_config),
                'candidates': [record['record'] for _, _, record in self.candidates],
                'initial_model_version': self.initial_model.version, 'best_trace': self.best_trace})

    @contextmanager
    def stage(self, name):
        self.phase = name
        lease = self.budgets.acquire(name)
        self.checkpoint()
        emit_event(self.context, 'stage_started', {'stage': name, 'round': self.budgets.started_rounds,
                                                  'deadline': lease.deadline})
        try:
            with isolated_solver_scope(self.context, cleanup_deadline=lease.deadline):
                yield lease
        finally:
            lease.close(cleanup_confirmed=not self.context.active_jobs)
            emit_event(self.context, 'stage_finished', {'stage': name, 'round': self.budgets.started_rounds,
                                                       'budgets': self.budgets.snapshot()})
            self.checkpoint()

    def lowering(self, inputs, lease):
        return LoweringContext(inputs, load_topology(inputs), self.model, lease, run_context=self.context)

    def store(self, bundle, inputs, origin):
        record = persist_candidate(self.context, bundle, inputs, origin=origin)
        self.candidates.append((bundle, inputs, record))
        emit_event(self.context, 'candidate_checked', {'origin': origin, 'xml_sha256': bundle.xml_sha256,
            'can_execute': bundle.can_execute, 'model_version': self.model.version,
            't_pred_us': None if bundle.prediction is None else bundle.prediction.t_pred_us,
            'record': record.get('xml')})
        self.checkpoint()
        return record

    def reassess(self, bundle, inputs, lease):
        refreshed = check_xml_bundle(bundle.xml_text, bundle.sidecar, self.lowering(inputs, lease))
        emit_event(self.context, 'candidate_reassessed', {'xml_sha256': bundle.xml_sha256,
            'xml_unchanged': refreshed.xml_sha256 == bundle.xml_sha256,
            'old_model_version': None if bundle.prediction is None else bundle.prediction.model_version,
            'model_version': self.model.version, 't_pred_us': None if refreshed.prediction is None else refreshed.prediction.t_pred_us})
        return refreshed

    def _initial_parameter_candidates(self, inputs, lowering, lease):
        history = inputs.request_config['sketch']['history']
        if history['schedule_reuse']:
            from vericcl.solver.history_index import find_schedules, adapt_schedule
            path = Path(history['schedule_index']).expanduser()
            if not path.is_absolute():
                path = self.context.root.parent.parent / path
            for hit in find_schedules(path, inputs, tolerance=history['relative_bandwidth_tolerance'],
                                      maximum=history['max_schedule_candidates'], lease=lease):
                lease.check()
                try:
                    bundle = lower_and_check(adapt_schedule(hit, inputs), lowering)
                except SemanticError:
                    continue
                yield bundle, 'history'
        if inputs.selected_synthesis_method == 'hierarchy':
            stream = iter(plan_hierarchy(inputs, lowering.topology, self.model, HierarchyContext(lease)))
        else:
            route_context = RouteContext(lease)
            stream = iter(optimize_fixed_routes(construct_routes(inputs, lowering.topology, self.model, route_context), route_context))
        try:
            for candidate in stream:
                lease.check()
                yield lower_and_check(candidate, lowering), 'initial'
        finally:
            if hasattr(stream, 'close'):
                stream.close()

    def initial_candidates(self):
        with self.stage('initial') as lease:
            domain = initial_parameter_domain(self.original, lease=lease)
            self.context.write_json('reports/initial-parameter-domain.json', plain(domain))
            pending, active = deque(), {}
            try:
                for ordinal, choice in enumerate(domain.candidates):
                    lease.check()
                    inputs = resolve_parameters(self.original, choice)
                    stream = iter(self._initial_parameter_candidates(inputs, self.lowering(inputs, lease), lease))
                    active[ordinal] = stream, inputs
                    pending.append(ordinal)
                while pending:
                    lease.check()
                    ordinal = pending.popleft()
                    stream, inputs = active[ordinal]
                    try:
                        bundle, origin = next(stream)
                        lease.check()
                    except StopIteration:
                        lease.check()
                    except BudgetExhausted as error:
                        if lease.expired():
                            raise
                        self.failures.append('initial parameter budget: ' + str(error))
                    except (SemanticError, ConstructionInfeasibleError) as error:
                        self.failures.append('initial candidate: ' + str(error))
                    else:
                        self.store(bundle, inputs, origin)
                        pending.append(ordinal)
                        continue
                    stream.close()
                    del active[ordinal]
            finally:
                # Closing while the phase lease is still active lets suspended
                # solver streams release their owned work before phase cleanup.
                for stream, _ in active.values():
                    stream.close()

    def calibration(self, inputs, *, only_modes=None):
        with self.stage('online') as lease:
            probe_inputs = _channel_cap(inputs, self.original.solver.max_channels)
            probes = ProbeContext(probe_inputs, load_topology(probe_inputs), lease, run_context=self.context,
                                  native_driver=self.services.probe_driver)
            jobs = None
            if only_modes is not None:
                jobs = [_mode(probes, mode[0].kind if len(mode) == 1 and mode[0].kind in {'copy', 'reduce'}
                              else 'link' if len(mode) == 1 else 'shared', mode, size)
                        for mode in only_modes for size in payload_sizes(probes)]
            result = run_calibration(probes, self.model, initial=self.initial_model, driver=self.services.probe_driver, jobs=jobs)
            self.model = result.model
            self.calibrated = True
            self.calibration_status = result.status
            self.failures.extend('calibration: ' + str(item.get('reason', 'unqualified component')) for item in result.issues)
            emit_event(self.context, 'calibration_finished', {'status': result.status, 'model_version': self.model.version,
                'issues': result.issues, 'observations': len(result.observations), 'confirmations': len(result.confirmations)})
            if result.status == 'cleanup_unconfirmed':
                raise ResourceCleanupError('calibration cleanup or GPU release is unconfirmed')

    def _native(self):
        if self.services.native_driver is None:
            try:
                from vericcl.live.native import NativeScheduleDriver
            except ImportError as error:
                raise SemanticError('native full-schedule execution driver is unavailable') from error
            self.services.native_driver = NativeScheduleDriver(self.context, self.original)
        return self.services.native_driver

    def measured_evidence(self, old, new, inputs):
        settings = self.original.request_config['sketch']['trace_and_measure']
        evidence = {'measured_required': settings['enabled']}
        if not settings['enabled']:
            return evidence
        from vericcl.xml.trace_sidecar import build_v5_trace_sidecar
        from vericcl.verification.online.trace_once import TraceContext, capture_once
        from vericcl.verification.online.trace_causality import analyze_trace_causality
        from vericcl.verification.online.trace_format import RAW_HEADER_STRUCT, RAW_RECORD_STRUCT
        from vericcl.verification.online.acceptance import AcceptanceContext, compare_measured
        from vericcl.verification.online.statistical_ledger import StatisticalLedger
        try:
            with self.stage('online') as lease:
                driver = self._native()
                trace_context = TraceContext(True, tuple(range(inputs.rank_count)),
                    driver.trace_driver(new, inputs, lease),
                    {rank: max(0, (settings['trace_capacity_bytes'] - RAW_HEADER_STRUCT.size) // RAW_RECORD_STRUCT.size)
                     for rank in range(inputs.rank_count)},
                    lease, self.context, warmup_iterations=settings['warmup_iterations'])
                trace = capture_once(build_v5_trace_sidecar(new), trace_context)
                evidence.update(trace_complete=trace.global_complete, trace_evidence_kind=trace.evidence_kind,
                                trace_status=trace.status, trace_issues=list(trace.issues))
                self.context.write_json('reports/trace-' + uuid.uuid4().hex + '.json', plain(trace))
                causal = analyze_trace_causality(new, trace, lease=lease)
                relative = self.context.relative_artifact('reports/trace-causality-' + (trace.capture_id or uuid.uuid4().hex) + '.json')
                path = self.context.write_json(relative, causal)
                evidence['trace_causality'] = {'path': relative, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'xml_sha256': new.xml_sha256, 'capture_id': trace.capture_id}
                if old is not None and trace.global_complete:
                    ledger = StatisticalLedger(self.context.artifact('statistics/attempts.jsonl'), task_id=self.context.task_id,
                        task_alpha=settings['task_alpha'], single_attempt_cap=settings['single_attempt_alpha_cap'], run_context=self.context)
                    acceptance = AcceptanceContext(True,
                        driver=lambda identity, launch: driver.measure(new if identity == new.xml_sha256 else old,
                            inputs if identity == new.xml_sha256 else self.best_inputs, launch),
                        seed=settings['seed'], minimum_gain=settings['min_relative_gain'], pair_count=settings['pairs'],
                        lease=lease, warmup=settings['warmup_iterations'], timed=settings['timed_iterations'],
                        max_rounds=settings['max_rounds'], max_cv=settings['max_cv'])
                    measured = compare_measured(old.xml_sha256, new.xml_sha256, ledger, acceptance)
                    evidence.update(measured_accepted=measured.accepted, measured=plain(measured))
        except (BudgetExhausted, SemanticError) as error:
            evidence.update(measured_accepted=False, failure=str(error))
            if 'trace_causality' not in evidence:
                evidence['trace_complete'] = False
            self.failures.append('trace/measurement: ' + str(error))
        return evidence

    def _read_trace(self, reference, bundle):
        from vericcl.verification.online.trace_causality import validate_trace_causality
        try:
            raw = self.context.artifact(reference['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != reference['sha256']:
                raise SemanticError('trace causal report checksum differs')
            report = json.loads(raw)
            validate_trace_causality(report, bundle)
            if reference['xml_sha256'] != report['xml_sha256'] or reference['capture_id'] != report['capture_id']:
                raise SemanticError('trace causal report reference identity differs')
            return report
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise SemanticError('trace causal report cannot be restored: ' + str(error)) from error

    def adopt(self, bundle, inputs, record, *, evidence=None):
        decision = accept_candidate(self.best, bundle, evidence or {'measured_required': False})
        if decision.accepted:
            reference = (evidence or {}).get('trace_causality')
            if reference is not None:
                self._read_trace(reference, bundle)
            elif (evidence or {}).get('measured_required'):
                raise SemanticError('accepted measured candidate lacks bound trace causal report')
        emit_event(self.context, 'candidate_decision', {'xml_sha256': bundle.xml_sha256, **plain(decision)})
        if decision.accepted:
            self.best, self.best_inputs, self.best_record = bundle, inputs, record
            self.best_trace = reference
            self.processed.clear(); self.expanded.clear(); self.analysis = None
            self.publish_best(decision.status)
            self.checkpoint()
        return decision.accepted

    def publish_best(self, reason):
        self.context.write_json('best/accepted.json', {'task_id': self.context.task_id,
            'xml_sha256': self.best.xml_sha256, 'model_version': self.best.prediction.model_version,
            't_pred_us': self.best.prediction.t_pred_us, 'reason': reason, 'candidate': self.best_record,
            'trace_causality': self.best_trace}, replace=True)

    def baseline(self):
        with self.stage('offline') as lease:
            viable = []
            for bundle, inputs, record in tuple(self.candidates):
                lease.check()
                if bundle.xml_text is None or bundle.checks['xml'].status.value == 'invalid':
                    continue
                if bundle.prediction is None or bundle.prediction.model_version != self.model.version:
                    bundle = self.reassess(bundle, inputs, lease)
                    record = self.store(bundle, inputs, 'model_reassessment')
                if bundle.can_execute:
                    viable.append((bundle, inputs, record))
            viable.sort(key=lambda item: Fraction(*item[0].prediction.exact_completion_ratio))
        for bundle, inputs, record in viable:
            evidence = self.measured_evidence(None, bundle, inputs)
            if self.adopt(bundle, inputs, record, evidence=evidence):
                return True
        return False

    def _capacity_recovery_requests(self, inputs, lease):
        size = inputs.hyperparameters.slice_size_bytes
        for choice in initial_parameter_domain(self.original, fixed_slice_bytes=size, lease=lease).candidates:
            yield ResynthesisRequest('structure', 'recover runtime capacity with a new legal schedule',
                                     channels=choice.channels)
        total = self.original.hyperparameters.total_size_bytes
        element = self.original.size_accounting['element_bytes']
        divided = self.original.collective.kind.value in {'alltoall', 'reduce_scatter'}
        # Larger complete slices are recovery proposals, not an assertion that
        # every capacity limit scales with S. Each proposal is lowered afresh.
        for count in range(inputs.hyperparameters.slice_count - 1, 0, -1):
            lease.check()
            if total % count or (divided and count % self.original.rank_count):
                continue
            proposed = total // count
            if proposed % element:
                continue
            for choice in initial_parameter_domain(self.original, fixed_slice_bytes=proposed, lease=lease).candidates:
                yield ResynthesisRequest('partition', 'recover an observed runtime capacity failure',
                                         proposed, choice.channels)

    def recover_capacity_baseline(self):
        if ((self.request.purpose == 'verify' and not self.request.tune)
                or self.budgets.started_rounds >= self.budgets.max_tuning_rounds
                or any(bundle.can_execute for bundle, _, _ in self.candidates)):
            return False
        required = {'semantic', 'storage', 'endpoint', 'deadlock', 'xml', 'prediction'}
        recoverable = [(bundle, inputs, record) for bundle, inputs, record in self.candidates
            if bundle.capacity_report is not None and bundle.capacity_report.violations
            and required <= bundle.checks.keys()
            and all(bundle.checks[key].status.value == 'valid' for key in required)]
        if not recoverable:
            return False
        self.budgets.start_tuning_round()
        emit_event(self.context, 'capacity_recovery_started', {'round': self.budgets.started_rounds,
            'candidate_count': len(recoverable), 'incumbent_present': False})
        with self.stage('repair') as lease:
            pending, active, started = deque(), {}, set()

            def advance(key):
                stream, record = active[key]
                lease.check()
                try:
                    proposal = next(stream, None)
                    # A solver can end its iterator after catching a budget
                    # exception. That does not complete the search branch.
                    lease.check()
                    if proposal is None:
                        self.expanded.add(key)
                    else:
                        self.store(proposal.candidate, proposal.context.inputs, 'capacity_recovery')
                        if proposal.candidate.can_execute:
                            emit_event(self.context, 'capacity_recovery_finished', {'recovered': True,
                                'source_record': record['record'], 'slice_bytes': proposal.context.inputs.hyperparameters.slice_size_bytes,
                                'channels': proposal.context.inputs.solver.max_channels,
                                'xml_sha256': proposal.candidate.xml_sha256})
                            return True
                        pending.append(key)
                        return False
                except BudgetExhausted:
                    if lease.expired():
                        raise
                except SemanticError as error:
                    if isinstance(error, SolverUnavailableError):
                        raise
                    self.failures.append('capacity recovery: ' + str(error))
                stream.close()
                del active[key]
                return False

            try:
                for invalid, inputs, record in recoverable:
                    repair = RepairContext(self.lowering(inputs, lease), request_inputs=self.original)
                    for branch in self._capacity_recovery_requests(inputs, lease):
                        key = 'capacity:%s:%s:%s:%s' % (inputs.hyperparameters.slice_size_bytes,
                            branch.kind, branch.slice_bytes, branch.channels)
                        if key in self.expanded or key in started:
                            continue
                        lease.check()
                        started.add(key)
                        active[key] = resynthesize_branch(invalid, branch, repair), record
                        # Try one complete seed per proposal first, then return
                        # to unfinished streams within this same repair lease.
                        if advance(key):
                            return True
                while pending:
                    if advance(pending.popleft()):
                        return True
            finally:
                for stream, _ in active.values():
                    stream.close()
        emit_event(self.context, 'capacity_recovery_finished', {'recovered': False,
            'scope': 'finite structural/partition branches; no infeasibility proof'})
        return False

    def _refine_candidate_model(self, bundle, inputs):
        if not self.original.request_config['sketch']['calibration']['enabled'] or bundle.executable is None:
            return
        if inputs.hyperparameters.slice_size_bytes != self.best_inputs.hyperparameters.slice_size_bytes:
            self.calibration(inputs)
            return
        modes = set()
        for mode in (() if bundle.prediction is None else bundle.prediction.unmeasured_modes):
            flows = []
            for item in mode:
                kind, src, dst, channel = item.split(':')
                flows.append(ProbeFlow(kind, int(src), int(dst), int(channel[1:])))
            modes.add(tuple(sorted(flows)))
        for operation in bundle.executable.operations:
            if operation.kind == 'nop':
                continue
            cost = self.model.service_for(operation.cost_key, operation.payload_bytes)
            if cost is not None and cost.source in {'measured', 'history'}:
                continue
            parts = operation.cost_key.split(':')
            kind = 'send' if parts[0] == 'link' else parts[0]
            src = int(parts[1]); dst = int(parts[2]) if len(parts) == 3 else src
            modes.add((ProbeFlow(kind, src, dst),))
        if modes:
            self.calibration(inputs, only_modes=tuple(sorted(modes)))

    def compare(self, bundle, inputs, record):
        preliminary = accept_candidate(self.best, bundle, {})
        can_calibrate_missing = (self.original.request_config['sketch']['calibration']['enabled'] and
            all(bundle.checks[key].status.value == 'valid' for key in ('semantic', 'storage', 'endpoint', 'deadlock', 'xml', 'launch')) and
            (bundle.prediction is None or bundle.prediction.t_pred_us is None))
        if not preliminary.accepted and not can_calibrate_missing:
            emit_event(self.context, 'candidate_decision', {'xml_sha256': bundle.xml_sha256, **plain(preliminary)})
            return False
        self._refine_candidate_model(bundle, inputs)
        if self.best.prediction.model_version != self.model.version or bundle.prediction.model_version != self.model.version:
            with self.stage('offline') as lease:
                self.best = self.reassess(self.best, self.best_inputs, lease)
                self.best_record = self.store(self.best, self.best_inputs, 'incumbent_model_reassessment')
                bundle = self.reassess(bundle, inputs, lease)
                record = self.store(bundle, inputs, 'candidate_model_reassessment')
                if self.best.can_execute:
                    self.publish_best('model_reassessment')
        if not accept_candidate(self.best, bundle, {}).accepted:
            return False
        return self.adopt(bundle, inputs, record, evidence=self.measured_evidence(self.best, bundle, inputs))

    def partition_benefit(self, branch, lease):
        from vericcl.solver.history_index import ScheduleHit, adapt_schedule
        from vericcl.tuning.strategies import _network
        choices = initial_parameter_domain(self.original, fixed_slice_bytes=branch.slice_bytes, lease=lease).candidates
        payload = {'slice_count': self.best_inputs.hyperparameters.slice_count,
                   'operations': plain(_network(self.best))}
        hit = ScheduleHit(payload, {rank: rank for rank in range(self.original.rank_count)}, 0.)
        for choice in choices:
            if choice.channels < self.best.program.channels:
                continue
            lease.check()
            inputs = resolve_parameters(self.original, choice)
            candidate = lower_and_check(adapt_schedule(hit, inputs), self.lowering(inputs, lease))
            record = self.store(candidate, inputs, 'partition_benefit_check')
            if accept_candidate(self.best, candidate, {}).accepted:
                emit_event(self.context, 'partition_benefit', {'slice_bytes': branch.slice_bytes,
                    'old_t_pred_us': self.best.prediction.t_pred_us, 'new_t_pred_us': candidate.prediction.t_pred_us,
                    'record': record['record'], 'action': 'full_initial_resynthesis_before_adoption'})
                return True
        return False

    def _expansion_requests(self):
        yield ResynthesisRequest('structure', reason='finite strategy generation exhausted; expand initial structures')
        size = self.original.hyperparameters.total_size_bytes
        element = self.original.size_accounting['element_bytes']
        maximum = min(size // element, 32768)
        for count in range(maximum, 0, -1):
            if size % count == 0:
                candidate = size // count
                if candidate != self.best_inputs.hyperparameters.slice_size_bytes:
                    yield ResynthesisRequest('partition', 'compare a complete resynthesis before changing S', candidate)

    def tuning(self):
        while True:
            if self.budgets.started_rounds >= self.budgets.max_tuning_rounds:
                return 'maximum_rounds'
            self.budgets.start_tuning_round()
            try:
                stop = self.tuning_round()
                if stop is not None:
                    return stop
            except BudgetExhausted as error:
                self.failures.append('round budget: ' + str(error))
                emit_event(self.context, 'round_budget_exhausted', {'round': self.budgets.started_rounds, 'message': str(error)})
                self.checkpoint()
                # The next start checks all cumulative and wall limits before
                # resetting only the next round's phase allowances.

    def tuning_round(self):
        with self.stage('offline') as lease:
            inputs = _channel_cap(self.best_inputs, self.original.solver.max_channels)
            lowering = self.lowering(inputs, lease)
            # A small proof attempt is prioritization, not an incomplete proof.
            proof_lease = StandaloneLease.after(min(1., max(.001, lease.remaining_seconds() / 10)))
            try:
                certificate = certify_global_cut(self.best, replace(lowering, lease=proof_lease))
            except BudgetExhausted:
                certificate = None
            if certificate and check_global_cut(certificate, self.best, lowering):
                self.proof = certificate
                self.context.write_json('reports/no-improvement-proof.json', certificate)
                return 'proven_no_improvement'
            if self.analysis is None:
                self.analysis = AnalysisContext(lowering, backend=self.original.request_config['sketch']['solver']['bdd_backend'],
                    trace_analysis=None if self.best_trace is None else self._read_trace(self.best_trace, self.best))
            self.analysis.lowering = lowering
            policy = self.original.request_config['sketch']['repair']
            batch = analyze(self.best, self.model, SearchScope('full-repair', batch_size=policy['analysis_batch_size']), self.analysis)
            emit_event(self.context, 'analysis_batch', plain(batch))
            if batch.status in {'analysis_error', 'model_unavailable'}:
                self.failures.append('analysis: ' + str(batch.evidence))
                return 'analysis_error'
            if batch.status == 'budget_exhausted':
                raise BudgetExhausted('offline analysis round budget exhausted')
            actions = tuple(action for action in batch.strategies if action.strategy_id not in self.processed)
        improved = False
        for action in actions:
            try:
                with self.stage('repair') as lease:
                    lowering = self.lowering(self.best_inputs, lease)
                    repair = RepairContext(lowering, mode=policy['mode'], candidate_batch_size=policy['candidate_batch_size'],
                                           request_inputs=self.original)
                    proposals = []
                    if action.family == 'structure':
                        stream = resynthesize_branch(self.best, ResynthesisRequest('structure', 'analyzed structure repair'), repair)
                        try:
                            for item in stream:
                                proposals.append((item.candidate, item.context.inputs,
                                                  self.store(item.candidate, item.context.inputs, 'structure_repair')))
                        finally:
                            stream.close()
                    else:
                        result = solve_repair(self.best, build_domain(None, action), action, repair)
                        emit_event(self.context, 'repair_finished', {'status': result.status, 'domain': plain(result.domain),
                            'attempts': result.attempts, 'solver_evidence': result.solver_evidence})
                        if result.candidate is not None:
                            candidate_inputs = result.context.inputs if result.context else self.best_inputs
                            proposals.append((result.candidate, candidate_inputs,
                                              self.store(result.candidate, candidate_inputs, 'local_repair')))
                    self.processed.add(action.strategy_id)
                for proposal in proposals:
                    if self.compare(*proposal):
                        improved = True
                        break
                if improved:
                    break
            except BudgetExhausted:
                raise
            except SemanticError as error:
                if isinstance(error, SolverUnavailableError):
                    raise
                self.processed.add(action.strategy_id)
                self.failures.append('local repair: ' + str(error))
                emit_event(self.context, 'repair_failed', {'strategy_id': action.strategy_id, 'message': str(error)})
        if not improved and batch.status == 'candidates_exhausted':
            emit_event(self.context, 'search_expanded', {'reason': 'finite candidates exhausted without a full-space proof'})
            for branch in self._expansion_requests():
                key = branch.kind + ':' + str(branch.slice_bytes)
                if key in self.expanded:
                    continue
                with self.stage('repair') as lease:
                    if branch.kind == 'partition' and not self.partition_benefit(branch, lease):
                        self.expanded.add(key)
                        continue
                    repair = RepairContext(self.lowering(self.best_inputs, lease), request_inputs=self.original)
                    stream = resynthesize_branch(self.best, branch, repair)
                    proposals = []
                    try:
                        for item in stream:
                            proposals.append((item.candidate, item.context.inputs,
                                              self.store(item.candidate, item.context.inputs, 'expanded_resynthesis')))
                    except SemanticError as error:
                        if isinstance(error, (BudgetExhausted, SolverUnavailableError)):
                            raise
                        self.failures.append('resynthesis: ' + str(error))
                    finally:
                        stream.close()
                    self.expanded.add(key)
                for proposal in proposals:
                    if self.compare(*proposal):
                        improved = True
                        break
                # One structural/S branch per outer round; the cursor persists.
                break
        self.phase = 'tuning'
        self.checkpoint()

    def run(self):
        if not self.restored:
            self.context.write_json('effective-request.json', plain(self.original.request_config))
            self.context.write_json('value-sources.json', plain(self.original.value_sources))
            self.context.write_json('size-accounting.json', plain(self.original.size_accounting))
            self.context.write_json('models/' + self.model.version + '.json', model_json(self.model))
        stop = 'unspecified'
        try:
            if self.restored and self.best_record is not None:
                with self.stage('offline') as lease:
                    self.best = check_xml_bundle(self.context.artifact(self.best_record['xml']).read_text(),
                        json.loads(self.context.artifact(self.best_record['sidecar']).read_text()), self.lowering(self.best_inputs, lease))
                    if not self.best.can_execute:
                        raise SemanticError('restored best fails current complete validation')
                    if self.best_trace is not None:
                        self._read_trace(self.best_trace, self.best)
                    elif self.original.request_config['sketch']['trace_and_measure']['enabled']:
                        raise SemanticError('restored measured best lacks bound trace causal report; reverify its XML in a new run')
                    self.publish_best('resume_revalidation')
            elif self.restored and self.restored.get('extra', {}).get('candidates'):
                with self.stage('offline') as lease:
                    for relative in self.restored['extra']['candidates']:
                        lease.check()
                        record = json.loads(self.context.artifact(relative).read_text())
                        config = record['effective_config']['sketch']
                        size = config['hyperparameters']['slice_size_bytes']
                        channels = config['solver']['max_channels']
                        choices = initial_parameter_domain(self.original, fixed_slice_bytes=size, lease=lease).candidates
                        choice = next((item for item in choices if item.channels == channels), None)
                        if choice is None or record['input_sha256'] != self.original.input_sha256 or 'xml' not in record:
                            continue
                        inputs = resolve_parameters(self.original, choice)
                        bundle = check_xml_bundle(self.context.artifact(record['xml']).read_text(),
                            json.loads(self.context.artifact(record['sidecar']).read_text()), self.lowering(inputs, lease))
                        self.candidates.append((bundle, inputs, record))
            elif self.request.purpose == 'verify':
                with self.stage('offline') as lease:
                    try:
                        inputs = bind_artifact_partition(self.original, self.request.sidecar)
                    except SemanticError as error:
                        inputs = self.original
                        bundle = rejected_xml_bundle(self.request.xml_text, self.request.sidecar, error)
                    else:
                        bundle = check_xml_bundle(self.request.xml_text, self.request.sidecar, self.lowering(inputs, lease))
                    self.store(bundle, inputs, 'verify_input')
            elif self.request.purpose == 'solve':
                try:
                    self.initial_candidates()
                except BudgetExhausted as error:
                    self.failures.append('initial synthesis: ' + str(error))
            if self.request.purpose == 'calibrate':
                choices = initial_parameter_domain(self.original).candidates
                if not choices:
                    raise SemanticError('calibration has no feasible integer partition')
                self.calibration(resolve_parameters(self.original, choices[0]))
                self.phase = 'finished'; self.checkpoint()
                qualified = any(piece.source in {'measured', 'history'} for pieces in self.model.services.values() for piece in pieces)
                complete = qualified and self.calibration_status == 'completed' and not self.failures
                return RunResult(complete, 'completed' if complete else 'incomplete',
                    model_coverage='qualified_components' if qualified else 'unqualified', acceptance='not_required',
                    stop_reason='calibration_finished', optional_failures=tuple(self.failures),
                    failure_category=None if complete else 'online')
            if self.best is None:
                if not self.candidates:
                    return RunResult(False, 'incomplete', stop_reason='no_initial_candidate',
                        failure_category='offline', optional_failures=tuple(self.failures))
                if self.original.request_config['sketch']['calibration']['enabled'] and not self.calibrated:
                    usable = next((item for item in self.candidates if item[0].xml_text is not None), None)
                    if usable:
                        self.calibration(usable[1])
                qualified = self.baseline()
                if not qualified and self.recover_capacity_baseline():
                    qualified = self.baseline()
                if not qualified:
                    return RunResult(False, 'incomplete', stop_reason='no_qualified_baseline',
                        failure_category='online' if self.original.request_config['sketch']['trace_and_measure']['enabled'] else 'invalid',
                        optional_failures=tuple(self.failures))
            if self.request.purpose == 'verify' and not self.request.tune:
                stop = 'verification_complete'
            else:
                stop = self.tuning()
        except BudgetExhausted as error:
            stop = 'time_limit'
            self.failures.append(str(error))
        except SolverUnavailableError:
            raise
        except ResourceCleanupError as error:
            self.failures.append(str(error))
            self.phase = 'finished'
            self.checkpoint()
            return RunResult(False, 'incomplete', stop_reason='cleanup_unconfirmed',
                optional_failures=tuple(self.failures), failure_category='online',
                acceptance='best_retained' if self.best is not None else 'not_evaluated')
        except StorageFailure:
            raise
        except SemanticError as error:
            self.failures.append(str(error))
            stop = 'execution_error'
        self.phase = 'finished'
        self.checkpoint()
        if self.best is None or not self.best.can_execute or self.best.prediction.model_version != self.model.version:
            return RunResult(False, 'incomplete', stop_reason=stop, optional_failures=tuple(self.failures), failure_category='offline')
        if stop in {'execution_error', 'analysis_error'}:
            return RunResult(False, 'failed', model_coverage=self.best.prediction.model_coverage, acceptance='best_retained',
                stop_reason=stop, optional_failures=tuple(self.failures), failure_category='offline' if stop == 'analysis_error' else 'internal')
        history = self.original.request_config['sketch']['history']
        if history['schedule_reuse']:
            from vericcl.solver.history_index import save_schedule
            path = Path(history['schedule_index']).expanduser()
            if not path.is_absolute():
                path = self.context.root.parent.parent / path
            save_schedule(path, self.best, self.best_inputs)
        self.context.write_json('reports/timing.json', self.budgets.snapshot() | {
            't_pred_us': self.best.prediction.t_pred_us, 'model_version': self.model.version,
            'measurement_ledger': self.context.relative_artifact('statistics/attempts.jsonl') if self.original.request_config['sketch']['trace_and_measure']['enabled'] else None}, replace=True)
        return RunResult(True, 'completed', checks={key: value.status.value for key, value in self.best.checks.items()
                        if key != 'compatibility' or value.status.value != 'not_run'},
            model_coverage=self.best.prediction.model_coverage, acceptance='accepted', stop_reason=stop,
            proof_scope=self.proof, optional_failures=tuple(self.failures))


def run_primary(request, context, *, services=None, restored=None, budgets=None):
    return PrimaryWorkflow(request, context, services, restored=restored, budgets=budgets).run()
