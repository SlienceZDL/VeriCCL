"""Shared admission of repair proposals (repair types 2 and 3).

A proposal is a complete network candidate that changes only operations of
its repair domain. Each one is lowered, repaired for correctness when needed
(type 1), checked for unchanged outside work, and finally compared with the
incumbent in the native event-column model under the current cost model.
"""
from fractions import Fraction

from vericcl.errors import SemanticError, SolverUnavailableError
from vericcl.execution.budget import BudgetExhausted
from vericcl.xml.lower import lower_and_check
from vericcl.xml.parser import check_xml_bundle
from .correctness import repair_correctness
from .event_columns import solve_event_columns
from .strategies import RepairResult, outside_invariants


def correctness_steps(context):
    inputs = getattr(context, 'request_inputs', None) or context.lowering.inputs
    config = (inputs.request_config or {}).get('sketch', {}).get('repair', {}) or {}
    return int((config.get('correctness') or {}).get('max_steps', 8))


def prepare_lowering(context):
    """Allow the selected S's largest feasible channel count (as solve_repair does)."""
    from dataclasses import replace
    if context.request_inputs is None:
        return
    from vericcl.planner.parameters import initial_parameter_domain, resolve_parameters
    parameters = initial_parameter_domain(context.request_inputs,
                                          fixed_slice_bytes=context.lowering.inputs.hyperparameters.slice_size_bytes,
                                          lease=context.lease)
    if parameters.candidates:
        upper = max(parameters.candidates, key=lambda candidate: candidate.channels)
        context.lowering = replace(context.lowering, inputs=resolve_parameters(context.request_inputs, upper))


def admit(candidate_source, incumbent, domain, context, attempts):
    """Lower one proposal; return an executable bundle inside the domain or None."""
    candidate = lower_and_check(candidate_source, context.lowering)
    if not candidate.can_execute:
        repaired, report = repair_correctness(candidate, context.lowering, max_steps=correctness_steps(context))
        attempts.append({'status': 'correctness_repair', 'result': report.as_json()})
        if repaired is None:
            attempts.append({'status': 'invalid_candidate', 'checks': {
                key: result.message for key, result in candidate.checks.items() if result.status.value == 'invalid'}})
            return None
        candidate = repaired
    external = outside_invariants(incumbent, candidate, domain)
    if not external.valid:
        attempts.append({'status': 'outside_changed', 'differences': external.differences})
        return None
    return candidate


def evaluate_proposals(bundle, domain, proposals, context, *, label):
    """Admit ``(description, network candidate)`` proposals and select the best."""
    initial_calls, attempts, evidence, admitted, seen = context.model_calls, [], [], [], set()
    unresolved = False

    def outcome(status, candidate=None):
        return RepairResult(status, bundle, candidate, domain, context.model_calls - initial_calls,
                            tuple(attempts), tuple(evidence), not unresolved, context=context.lowering)
    try:
        context.lease.check()
        incumbent = check_xml_bundle(bundle.xml_text, bundle.sidecar, context.lowering)
        if not incumbent.can_execute:
            return outcome('incumbent_unavailable')
        for description, proposal in proposals:
            context.lease.check()
            attempts.append({'status': 'proposal', 'family': label, 'proposal': description})
            candidate = admit(proposal, incumbent, domain, context, attempts)
            if candidate is None or candidate.xml_sha256 in seen or candidate.xml_sha256 == incumbent.xml_sha256:
                continue
            seen.add(candidate.xml_sha256)
            admitted.append(candidate)
        if not admitted:
            return outcome('no_valid_proposal' if attempts else 'no_proposal')
        best = None
        for start in range(0, len(admitted), context.candidate_batch_size):
            columns = (incumbent,) + tuple(admitted[start:start + context.candidate_batch_size])
            try:
                selected, result = solve_event_columns(columns, context)
            except BudgetExhausted:
                if context.lease.expired():
                    raise
                unresolved = True
                evidence.append({'status': 'model_budget_exhausted', 'columns': len(columns)})
                continue
            evidence.append({'status': result.status, 'solver_status': result.solver_status, 'primal': result.primal,
                             'dual_bound': result.dual_bound, 'columns': len(columns), 'family': label})
            if not result.numerical_optimal:
                unresolved = True
            if selected is not None and selected > 0:
                chosen = columns[selected]
                ratio = Fraction(*chosen.prediction.exact_completion_ratio)
                if ratio < Fraction(*incumbent.prediction.exact_completion_ratio) and (
                        best is None or ratio < Fraction(*best.prediction.exact_completion_ratio)):
                    best = chosen
        return outcome('improved', best) if best is not None else outcome(
            'search_incomplete' if unresolved else 'no_improvement')
    except BudgetExhausted as error:
        attempts.append({'status': 'budget_exhausted', 'message': str(error)})
        return outcome('budget_exhausted')
    except SolverUnavailableError:
        raise
    except SemanticError as error:
        attempts.append({'status': 'solver_or_model_error', 'message': str(error)})
        return outcome('analysis_error')
