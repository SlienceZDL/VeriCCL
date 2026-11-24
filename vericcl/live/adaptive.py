"""Foreground request boundaries for one background repair task per group."""
import time

from vericcl.execution.budget import BudgetExhausted, ResourceBusy, StandaloneLease
from vericcl.execution.events import emit_event
from vericcl.live.background import BackgroundWorker
from vericcl.live.external import ExternalTelemetryFeed
from vericcl.live.model import update_live_model
from vericcl.live.telemetry import summarize_request
from vericcl.live.triggers import LiveTrigger
from vericcl.live.tuning import LiveRepairEngine, used_links
from vericcl.topology import load_topology
from vericcl.verification.online.model_publication import model_json
from vericcl.workflow_state import plain, persist_candidate
from vericcl.workflow_v5 import accept_candidate
from vericcl.xml.lower import LoweringContext
from vericcl.xml.v5_lower import evaluate_program


class AdaptiveLiveController:
    def __init__(self, session, workflow, *, validator=None):
        self.session, self.context, self.workflow = session, session.context, workflow
        self.inputs, self.bundle, self.model = workflow.best_inputs, workflow.best, workflow.model
        self.original = workflow.original; self.settings = session.settings
        from vericcl.live.validation import LiveBoundaryValidator
        self.validator = validator or LiveBoundaryValidator(session, current_inputs=lambda: self.inputs)
        self.invalid_links = set(); self.request_id = 0; self.recovery_deadline = None
        self.pending_proposal = None; self.last_failure = None
        self.trigger = LiveTrigger(**{key: self.settings[key] for key in (
            'window_requests', 'consecutive_windows', 'relative_change_threshold', 'min_residence_requests', 'baseline_max_cv')})
        self.trigger.version = self.bundle.xml_sha256
        self.worker = BackgroundWorker(LiveRepairEngine(self.context, workflow.budgets),
            timeout_s=self.settings['background_tuning_task_timeout_s'], parent_lease=session.lease)
        topology = load_topology(self.original)
        path = self.settings.get('external_telemetry_path')
        self.external = None if path is None else ExternalTelemetryFeed(path, group_id=self.context.task_id,
            legal_links={(key.src_rank, key.dst_rank) for key in topology.links}, max_age_s=self.settings['external_max_age_s'])

    def snapshot(self):
        return dict(version=self.bundle.xml_sha256, bundle=self.bundle, inputs=self.inputs, original=self.original,
                    model=self.model, model_updated=True, invalid_links=tuple(sorted(self.invalid_links)))

    def submit(self, event):
        if event.version != self.bundle.xml_sha256:
            raise ValueError('live event belongs to a different current version')
        self.invalid_links.update(event.invalid_links)
        previous = self.model.version
        self.model = update_live_model(self.model, (event,), payload_bytes=self.inputs.hyperparameters.slice_size_bytes)
        if self.model.version != previous:
            self.trigger.rebase_model()
            self.context.write_json('models/' + self.model.version + '.json', model_json(self.model))
        if used_links(self.bundle) & self.invalid_links and self.recovery_deadline is None:
            self.recovery_deadline = time.monotonic() + min(self.settings['recovery_timeout_s'], self.session.lease.remaining_seconds())
        emit_event(self.context, 'live_trigger', {'event': plain(event), 'model_version': self.model.version,
                                                'invalid_links': plain(self.invalid_links)})
        self.worker.offer(self.snapshot(), event)

    def _poll_external(self):
        if self.external is not None:
            for event in self.external.poll(version=self.bundle.xml_sha256, request_id=self.request_id):
                self.submit(event)

    def observe(self, receipt):
        summary = summarize_request(self.context.task_id, receipt, ranks=self.inputs.rank_count)
        if summary.version != self.bundle.xml_sha256:
            raise ValueError('completed request differs from the prepared live version')
        self.request_id = summary.request_id
        self.context.write_json('live/telemetry-%06d.json' % summary.request_id, plain(summary))
        event = self.trigger.observe_request(summary)
        if event is not None:
            self.submit(event)

    def _commit_proposal(self, value):
        if value.bundle is None:
            self.last_failure = value.status
            return False
        if value.model.version != self.model.version:
            self.last_failure = 'stale_model_proposal'
            emit_event(self.context, 'live_proposal_stale', {'proposal_model': value.model.version, 'current_model': self.model.version})
            return False
        fault = bool(used_links(self.bundle) & self.invalid_links)
        if not fault and self.trigger.last_switch is not None and self.trigger.successes - self.trigger.last_switch < self.trigger.residence:
            self.pending_proposal = value
            return False
        try:
            lease = self.workflow.budgets.acquire('offline')
        except ResourceBusy:
            self.pending_proposal = value
            return False
        except BudgetExhausted:
            self.last_failure = 'revalidation_budget_exhausted'
            self.pending_proposal = None
            return False
        lease.parent = self.session.lease
        lease.deadline = min(lease.deadline, self.session.lease.deadline,
                             self.recovery_deadline or float('inf'))
        lease.work_deadline = min(lease.work_deadline, lease.deadline - self.workflow.budgets.cleanup_reserve_s)
        try:
            lease.check()
            new_lowering = LoweringContext(value.inputs, load_topology(value.inputs), self.model, lease, run_context=self.context)
            new = evaluate_program(value.bundle.source, value.bundle.storage_plan, value.bundle.program, value.bundle.xml_text, new_lowering)
            old_lowering = LoweringContext(self.inputs, load_topology(self.inputs), self.model, lease, run_context=self.context)
            old = evaluate_program(self.bundle.source, self.bundle.storage_plan, self.bundle.program, self.bundle.xml_text, old_lowering)
        except BudgetExhausted:
            self.last_failure = 'revalidation_budget_exhausted'
            self.pending_proposal = None
            return False
        finally:
            # Reassessment is synchronous CPU work and creates no worker job.
            lease.close(cleanup_confirmed=True)
        if used_links(new) & self.invalid_links or not accept_candidate(None if fault else old, new, {}).accepted:
            self.last_failure = 'latest_state_validation_rejected'
            return False
        evidence = {'measured_required': self.original.request_config['sketch']['trace_and_measure']['enabled']}
        if evidence['measured_required']:
            online = self.workflow.budgets.acquire('online')
            online.parent = self.session.lease
            online.deadline = min(online.deadline, self.session.lease.deadline,
                                  self.recovery_deadline or float('inf'))
            online.work_deadline = min(online.work_deadline, online.deadline - self.workflow.budgets.cleanup_reserve_s)
            complete = False
            try:
                online.check()
                evidence = self.validator(None if fault else old, new, value.inputs, online)
                complete = True
            finally:
                online.close(cleanup_confirmed=complete and self.session.client.pending is None)
            if not accept_candidate(None if fault else old, new, evidence).accepted:
                self.last_failure = 'required_gpu_acceptance_rejected'
                return False
        lease = StandaloneLease.after(min(self.settings['version_prepare_timeout_s'], self.session.lease.remaining_seconds()))
        if self.recovery_deadline is not None:
            lease.deadline = min(lease.deadline, self.recovery_deadline)
        lease.check()
        receipt = self.session.prepare_version(new, value.inputs, lease=lease)
        if receipt.get('prepared') is not True or receipt.get('activated') is not False or receipt.get('version_id') != new.xml_sha256:
            raise RuntimeError('version preparation lacks a unanimous immutable READY receipt')
        previous = self.bundle.xml_sha256
        self.bundle, self.inputs = new, value.inputs
        self.trigger.switched(new.xml_sha256, successful_request=self.request_id)
        self.pending_proposal = None; self.recovery_deadline = None; self.last_failure = None
        record = persist_candidate(self.context, new, value.inputs, origin='live_fault_replacement' if fault else 'live_performance_update')
        self.context.write_json('live/prepared-%06d-%s.json' % (self.request_id, new.xml_sha256),
            {'previous': previous, 'next_version': new.xml_sha256, 'next_request_after': self.request_id,
             'model_version': self.model.version, 'reason': value.status, 'invalid_links': plain(self.invalid_links),
             'evidence': evidence, 'candidate': record, 'activation': 'next_unanimous_request_commit'})
        emit_event(self.context, 'live_version_prepared', {'version': new.xml_sha256, 'previous': previous, 'fault': fault})
        return True

    def _consume(self):
        result = self.worker.take_completed()
        if result is not None:
            if self.worker.valid_for(result, self.bundle.xml_sha256):
                self._commit_proposal(result.value)
            else:
                self.last_failure = result.error or 'stale_version_proposal'
                emit_event(self.context, 'live_background_rejected', {'reason': self.last_failure})
            self.worker.start_pending(self.snapshot())
        if self.pending_proposal is not None:
            self._commit_proposal(self.pending_proposal)

    def before_request(self):
        self.session.lease.check(); self._poll_external(); self._consume()
        while used_links(self.bundle) & self.invalid_links:
            if self.worker.idle or time.monotonic() >= self.recovery_deadline:
                raise RuntimeError('no qualified legal replacement within the fixed recovery budget: ' + str(self.last_failure))
            self.worker.join(min(.02, max(0, self.recovery_deadline - time.monotonic())))
            self._poll_external(); self._consume()

    def close(self, seconds):
        complete = self.worker.close(seconds)
        self.context.write_json('live/background-close.json', {'cleanup_complete': complete,
            'invalid_links': plain(self.invalid_links), 'last_failure': self.last_failure}, replace=True)
        return complete
