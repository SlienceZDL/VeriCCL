from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from math import isfinite
from time import monotonic
from typing import Dict, Mapping, Tuple

from vericcl.errors import SemanticError
from vericcl.input.models import ResolvedInput
from vericcl.semantics.atom import Schedule
from vericcl.solver.budget import ModelBudget
from vericcl.solver.scip_api import ScipAdapter
from vericcl.topology.model import LaneKey, Topology
from vericcl.tuning.impact import ImpactClosure
from vericcl.tuning.model import RepairResult, RepairStatus, TuningOverlay
from vericcl.tuning.repair import (
    _apply_timing, _legal_options, _rewrite_flow_suffix, _timing_dependencies,
)
from vericcl.verification.bdd_flow import FlowReplacementHint
from vericcl.verification.constraints import verify_schedule_constraints
from vericcl.verification.flow_index import build_flow_index
from vericcl.verification.model import ValidationStatus
from vericcl.verification.semantics import verify_schedule_semantics
from vericcl.xml.lower import lower_to_xml


MAX_TRANSFERS = 256
MAX_ORDER_PAIRS = 4096
PROOF_ABSOLUTE_TOLERANCE_US = 1e-9
PROOF_RELATIVE_TOLERANCE = 1e-9


def _result(status: RepairStatus, code: str, message: str, **evidence):
    values = dict(evidence)
    values.update({"scope": "local", "implementation": "local_timing_v2",
                   "code": code, "message": message})
    return RepairResult(status, None, frozenset(), None, "local_milp", values)


def _ordered_queues(schedule: Schedule) -> Tuple[Tuple[str, ...], ...]:
    raw_slots = schedule.metadata.get("resource_slots", {})
    if not isinstance(raw_slots, Mapping):
        raise SemanticError("resource_slots must be a mapping")
    queues: Dict[object, list] = {}
    for transfer in schedule.transfers:
        lane = LaneKey(transfer.src_rank, transfer.dst_rank, transfer.channel)
        queues.setdefault(("lane", lane), []).append(transfer)
        slots = raw_slots.get(transfer.transfer_id, {})
        if not isinstance(slots, Mapping):
            raise SemanticError("resource slot assignment must be a mapping")
        for resource_id, slot in slots.items():
            queues.setdefault(("resource", resource_id, slot), []).append(transfer)
    result = []
    for values in queues.values():
        values.sort(key=lambda item: (item.st_time, item.ed_time, item.transfer_id))
        result.append(tuple(item.transfer_id for item in values))
    return tuple(result)


def _semantic_dependencies(schedule: Schedule) -> Mapping[str, frozenset[str]]:
    raw = schedule.metadata.get("semantic_predecessors", {})
    if not isinstance(raw, Mapping):
        raise SemanticError("semantic_predecessors must be a mapping")
    known = {transfer.transfer_id for transfer in schedule.transfers}
    result = {}
    for transfer in schedule.transfers:
        try:
            values = transfer.predecessor_ids | frozenset(raw.get(transfer.transfer_id, ()))
        except TypeError as error:
            raise SemanticError("semantic predecessor IDs must be iterable") from error
        if not values <= known:
            raise SemanticError("semantic predecessor is missing")
        result[transfer.transfer_id] = values
    return result


def _structure(transfer):
    return (transfer.src_rank, transfer.dst_rank, transfer.channel,
            transfer.stage_id, transfer.kind, transfer.member_slice_ids,
            transfer.predecessor_ids, tuple(
                (atom.slice_id, tuple((stage.stage_id, stage.operator, tuple(
                    (symbol.src_rank, symbol.dst_rank) for symbol in stage.symbols
                )) for stage in atom.path)) for atom in transfer.atoms
            ))


def _unsupported_restrictions(schedule, overlay, inputs):
    if (overlay.lane_order or overlay.tree_roots or overlay.tree_edges
            or overlay.batch_size is not None or overlay.hierarchy_template
            or overlay.resolve_scope or inputs.strategies.manual_hierarchy):
        return True
    restricted = {"timing_constraints", "manual_timing", "lane_order",
                  "fixed_start_times", "fixed_end_times", "manual_order"}
    return any(key in values and values[key] for values in
               (schedule.metadata, inputs.resolved_atom, inputs.resolved_sketch)
               for key in restricted)


def solve_local_repair(
    schedule: Schedule,
    hint: FlowReplacementHint,
    impact: ImpactClosure,
    overlay: TuningOverlay,
    topology: Topology,
    inputs: ResolvedInput,
    budget: ModelBudget,
) -> RepairResult:
    started = monotonic()
    for value, expected, name in (
        (schedule, Schedule, "schedule"), (hint, FlowReplacementHint, "hint"),
        (impact, ImpactClosure, "impact"), (overlay, TuningOverlay, "overlay"),
        (topology, Topology, "topology"), (inputs, ResolvedInput, "inputs"),
        (budget, ModelBudget, "budget"),
    ):
        if not isinstance(value, expected):
            raise SemanticError("{} must be a {}".format(name, expected.__name__))
    parent = {t.transfer_id: t for t in schedule.transfers}
    if not impact.transfer_ids <= set(parent):
        raise SemanticError("impact closure contains an unknown transfer")
    evidence = {"scope": "local", "implementation": "local_timing_v2",
                "affected_transfer_count": len(impact.transfer_ids),
                "input_impact_transfer_ids": tuple(sorted(impact.transfer_ids)),
                "build_seconds": 0.0, "optimize_seconds": 0.0,
                "extract_seconds": 0.0, "recheck_seconds": 0.0}

    def fail(status, code, message):
        evidence["total_seconds"] = monotonic() - started
        return _result(status, code, message, **evidence)

    if budget.seconds <= 0.0:
        return fail(RepairStatus.TIMEOUT, "local_repair_budget_exhausted", "budget exhausted")
    if not hint.candidate_flow_ids:
        return fail(RepairStatus.INFEASIBLE, "local_repair_has_no_candidate", "no candidate flow")
    try:
        overlay.validate_against(inputs, schedule, topology)
        _semantic_dependencies(schedule)
        parent_queues = _ordered_queues(schedule)
        if _unsupported_restrictions(schedule, overlay, inputs):
            return fail(RepairStatus.NOT_RUN, "local_repair_unsupported_restrictions",
                        "manual timing, order, tree, or hierarchy restrictions are unsupported")
        flow = build_flow_index(schedule).flow(hint.source_flow_id)
        if flow.demand_id != hint.demand_id or hint.divergence_rank not in flow.ranks:
            raise SemanticError("hint does not match source flow")
        divergence = flow.ranks.index(hint.divergence_rank)
        if divergence >= flow.comparison_end or flow.transfer_ids[divergence] != hint.waiting_transfer_id:
            raise SemanticError("hint waiting transfer does not match source flow")
        options = _legal_options(flow, hint, overlay, topology, inputs, schedule)
        evidence["legal_candidate_count"] = len(options)
        evidence["candidate_comparison_scope"] = "one deterministic path-cost-ranked strategy; not minimax load scoring"
        if not options:
            return fail(RepairStatus.INFEASIBLE, "local_repair_has_no_legal_candidate", "no legal candidate flow")
        selected = options[0].candidate_id
        evidence["selected_candidate_flow_id"] = selected
        narrowed = replace(hint, candidate_flow_ids=(selected,),
                           candidate_paths={selected: hint.candidate_paths[selected]},
                           candidate_first_lanes={selected: hint.candidate_first_lanes[selected]})
        draft_result = _rewrite_flow_suffix(schedule, narrowed, overlay, topology,
                                            inputs, schedule_and_validate=False)
        if draft_result.status is not RepairStatus.SUCCESS:
            evidence["structural_failure"] = dict(draft_result.evidence)
            return fail(draft_result.status, "local_repair_selected_strategy_invalid", "selected strategy could not be rewritten")
        draft = draft_result.schedule
        evidence.update(dict(draft_result.evidence))
        by_id = {t.transfer_id: t for t in draft.transfers}
        if len(by_id) > MAX_TRANSFERS:
            return fail(RepairStatus.NOT_RUN, "local_repair_model_size_limit", "bounded timing model size exceeded")
        semantic = _semantic_dependencies(draft)
        operations, dependencies = _timing_dependencies(draft.transfers, semantic)
        parent_semantic = _semantic_dependencies(schedule)
        domain = (impact.transfer_ids & set(by_id)) | {
            key for key, transfer in by_id.items()
            if key not in parent or _structure(transfer) != _structure(parent[key])
            or semantic[key] != parent_semantic[key]
        }
        outside = set(by_id) - domain
        pairs = {tuple(sorted(pair)) for queue in _ordered_queues(draft)
                 for pair in combinations(queue, 2)}
        local_pairs = sorted(pair for pair in pairs if set(pair) & domain)
        fixed_pairs = set()
        for queue in parent_queues:
            fixed = tuple(key for key in queue if key in outside)
            fixed_pairs.update(zip(fixed, fixed[1:]))
        evidence.update({
            "modeled_transfer_ids": tuple(sorted(by_id)),
            "decision_domain_transfer_ids": tuple(sorted(domain)),
            "outside_transfer_ids": tuple(sorted(outside)),
            "fixed_transfer_count": len(outside),
            "preserved_outside_order_pairs": tuple(sorted(fixed_pairs)),
            "outside_paths_channels_preserved": True,
            "timing_propagation_transfer_count": len(by_id),
            "order_decision_count": len(local_pairs),
            "optimality_scope": "fixed selected route/channel/resource slots; local order; outside starts lower-bounded",
        })
        if len(by_id) > MAX_TRANSFERS or len(local_pairs) > MAX_ORDER_PAIRS:
            return fail(RepairStatus.NOT_RUN, "local_repair_model_size_limit", "bounded timing model size exceeded")
    except (SemanticError, KeyError, TypeError, ValueError) as error:
        return fail(RepairStatus.INVALID, "local_repair_invalid", str(error))

    model = None
    try:
        gp = ScipAdapter.require()
        if "final_outputs" not in schedule.metadata:
            return fail(RepairStatus.NOT_RUN, "local_repair_incomplete_collective_contract", "full final-output contract is required")
        from vericcl.execution.budget import StandaloneLease, available_cpu_slots
        import math
        gp, model = ScipAdapter.create_model("vericcl-local-timing")
        model_lease = StandaloneLease(budget.deadline)
        ScipAdapter.configure(model, seed=inputs.solver.solver_seed,
                              threads=min(inputs.solver.max_threads_per_model, available_cpu_slots()),
                              gap=0.0 if inputs.solver.require_proven_optimal else inputs.solver.mip_gap,
                              lease=model_lease)
        parameter_names = {"MIPGap": "limits/gap", "MIPGapAbs": "limits/absgap",
                           "FeasibilityTol": "numerics/feastol"}
        for name, value in overlay.milp_parameters:
            if name in {"TimeLimit", "Seed", "Threads"}:
                continue
            if name not in parameter_names:
                raise SemanticError("unsupported local SCIP parameter: " + name)
            model.setParam(parameter_names[name], value)
        if inputs.solver.require_proven_optimal:
            model.setParam("limits/gap", 0.0)
            model.setParam("limits/absgap", 0.0)
        evidence.update({"effective_mip_gap_limit": model.getParam("limits/gap"),
                         "effective_mip_gap_abs_limit": model.getParam("limits/absgap")})
        starts, ends = {}, {}
        for index, key in enumerate(sorted(by_id)):
            transfer = by_id[key]
            release = max((atom.path[0].symbols[0].ready_time for atom in transfer.atoms), default=0.0)
            lower = max(release, parent[key].st_time) if key in outside else release
            starts[key] = model.addVar(lb=lower, name="start_{}".format(index))
            ends[key] = model.addVar(lb=0.0, name="end_{}".format(index))
            model.addCons(ends[key] == starts[key] + transfer.ed_time - transfer.st_time,
                            name="duration_{}".format(index))
        for successor, predecessors in dependencies.items():
            for predecessor in sorted(predecessors):
                model.addCons(starts[successor] >= ends[predecessor])
        for left, right in sorted(fixed_pairs):
            model.addCons(starts[right] >= ends[left])
        for index, (left, right) in enumerate(local_pairs):
            order = model.addVar(vtype="B", name="order_{}".format(index))
            ScipAdapter.indicator(model, order, True, starts[right] >= ends[left])
            ScipAdapter.indicator(model, order, False, starts[left] >= ends[right])
        makespan = model.addVar(lb=0.0, name="makespan")
        for end in ends.values():
            model.addCons(makespan >= end)
        model.setObjective(makespan, "minimize")
        # Native SCIP variables and constraints are registered immediately.
        evidence.update({"timing_variable_count": len(starts) + len(ends),
                         "model_variable_count": model.getNVars(),
                         "model_constraint_count": model.getNConss(),
                         "model_general_constraint_count": model.general_constraint_count(),
                         "build_seconds": monotonic() - started})
        remaining = budget.seconds - (monotonic() - started)
        if remaining <= 0.0:
            return fail(RepairStatus.TIMEOUT, "local_repair_budget_exhausted", "budget exhausted while building model")
        model_lease.check()
        optimize_started = monotonic()
        model.optimize()
        evidence["optimize_seconds"] = monotonic() - optimize_started
        evidence["solver_status"] = model.result.status
        evidence["solution_count"] = int(model.getNSols())
        if model.getNSols() == 0:
            status = (RepairStatus.TIMEOUT if model.result.status == "time_limit" else
                      RepairStatus.INFEASIBLE if model.result.status == "infeasible" else RepairStatus.INVALID)
            return fail(status, "local_repair_has_no_solution", "selected-strategy timing model has no incumbent")
        objective, bound = float(model.result.primal), float((0.0 if model.result.dual_bound is None else model.result.dual_bound))
        absolute_gap = abs(objective - bound)
        relative_gap = (absolute_gap / abs(objective) if objective != 0.0 else
                        0.0 if absolute_gap == 0.0 else float("inf"))
        reported_gap = relative_gap if model.result.relative_gap is None else float(model.result.relative_gap)
        # This is numerical model optimality, with both tests required. The tiny
        # absolute tolerance cannot grow with objective scale; the relative test
        # also rejects material gaps for near-zero objectives.
        proven = (model.result.status == "optimal"
                  and all(isfinite(value) for value in (objective, bound, reported_gap))
                  and absolute_gap <= PROOF_ABSOLUTE_TOLERANCE_US
                  and relative_gap <= PROOF_RELATIVE_TOLERANCE
                  and 0.0 <= reported_gap <= PROOF_RELATIVE_TOLERANCE)
        evidence.update({"objective_us": objective, "objective_bound_us": bound,
                         "objective_gap": reported_gap,
                         "objective_absolute_gap_us": absolute_gap,
                         "objective_relative_bound_gap": relative_gap,
                         "proof_absolute_tolerance_us": PROOF_ABSOLUTE_TOLERANCE_US,
                         "proof_relative_tolerance": PROOF_RELATIVE_TOLERANCE,
                         "proven_optimal": proven})
        if inputs.solver.require_proven_optimal and not proven:
            return fail(RepairStatus.TIMEOUT, "local_repair_optimality_required", "incumbent lacks required optimality proof")
        extraction_started = monotonic()
        start_values = {key: max(0.0, float(model.getVal(value))) for key, value in starts.items()}
        end_values = {key: float(model.getVal(value)) for key, value in ends.items()}
        transfers = _apply_timing(draft.transfers, operations, start_values, end_values)
        transfers = tuple(replace(t, predecessor_ids=dependencies[t.transfer_id]) for t in transfers)
        metadata = dict(draft.metadata)
        metadata["semantic_predecessors"] = {key: tuple(sorted(value)) for key, value in dependencies.items()}
        repaired = replace(draft, transfers=transfers, metadata=metadata)
        evidence["extract_seconds"] = monotonic() - extraction_started
    except Exception as error:
        evidence.update({"error_type": type(error).__name__, "error": str(error)})
        return fail(RepairStatus.NOT_RUN, "local_repair_solver_unavailable", "solver or model extraction failed")
    finally:
        if model is not None:
            model.freeProb()

    recheck_started = monotonic()
    forbidden = tuple(inputs.atom_constraints.forbidden_transfers) + tuple(overlay.temporary_forbidden)
    violation = any(item.slice_id in t.member_slice_ids and item.stage_id == t.stage_id
                    and item.src_rank == t.src_rank and item.dst_rank == t.dst_rank
                    for item in forbidden for t in repaired.transfers)
    constraints = verify_schedule_constraints(repaired, inputs, topology)
    semantics = verify_schedule_semantics(repaired, inputs)
    evidence.update({"constraint_status": constraints.status.value,
                     "semantic_status": semantics.status.value,
                     "recheck_seconds": monotonic() - recheck_started,
                     "constraint_code": constraints.code, "semantic_code": semantics.code})
    if violation or constraints.status is not ValidationStatus.VALID or semantics.status is not ValidationStatus.VALID:
        return fail(RepairStatus.INVALID, "local_repair_merged_validation_failed", "merged schedule failed full validation")
    parent_makespan = max((t.ed_time for t in schedule.transfers), default=0.0)
    repaired_makespan = max((t.ed_time for t in repaired.transfers), default=0.0)
    evidence.update({"parent_makespan_us": parent_makespan, "repaired_makespan_us": repaired_makespan})
    if repaired_makespan >= parent_makespan - 1e-8:
        return fail(RepairStatus.NOT_RUN, "local_repair_no_improvement", "selected strategy did not improve makespan")
    try:
        artifact = lower_to_xml(repaired, inputs, topology)
        effective = artifact.buffer_plan.transfer_effective_times
        mismatches = tuple(sorted(t.transfer_id for t in repaired.transfers
                                  if effective[t.transfer_id] != (t.st_time, t.ed_time)))
        evidence.update({"buffer_status": "valid", "xml_status": "valid",
                         "deadlock_status": "valid", "xml_sha256": artifact.sha256,
                         "runtime_compatible": artifact.runtime_compatible,
                         "buffer_retimed_transfer_ids": mismatches})
        if mismatches:
            evidence["recheck_seconds"] = monotonic() - recheck_started
            return fail(RepairStatus.NOT_RUN, "local_repair_buffer_schedule_mismatch",
                        "buffer lowering changes solved times; selected strategy rejected")
    except SemanticError as error:
        evidence["recheck_seconds"] = monotonic() - recheck_started
        evidence["lowering_error"] = str(error)
        return fail(RepairStatus.INVALID, "local_repair_lowering_failed", "buffer/XML validation failed")
    evidence["recheck_seconds"] = monotonic() - recheck_started
    changed = (set(parent) - set(by_id)) | {t.transfer_id for t in repaired.transfers if parent.get(t.transfer_id) != t}
    evidence["total_seconds"] = monotonic() - started
    return RepairResult(RepairStatus.SUCCESS, repaired, frozenset(changed), selected, "local_milp", evidence)
