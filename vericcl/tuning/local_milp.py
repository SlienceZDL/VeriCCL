from __future__ import annotations

from dataclasses import replace
from typing import Dict, Mapping, Tuple

from vericcl.errors import SemanticError
from vericcl.input.models import ResolvedInput
from vericcl.semantics.atom import Schedule
from vericcl.solver.budget import ModelBudget
from vericcl.solver.gurobi_api import GurobiAdapter
from vericcl.topology.model import LaneKey, Topology
from vericcl.tuning.impact import ImpactClosure
from vericcl.tuning.model import RepairResult, RepairStatus, TuningOverlay
from vericcl.tuning.repair import _legal_options, repair_flow_suffix
from vericcl.verification.bdd_flow import FlowReplacementHint
from vericcl.verification.flow_index import build_flow_index


def _result(status: RepairStatus, code: str, message: str, **evidence):
    values = dict(evidence)
    values.update({"scope": "local", "code": code, "message": message})
    return RepairResult(
        status=status,
        schedule=None,
        changed_transfer_ids=frozenset(),
        selected_candidate_flow_id=None,
        method="local_milp",
        evidence=values,
    )


def _ordered_queues(
    schedule: Schedule,
) -> Tuple[Tuple[str, ...], ...]:
    raw_slots = schedule.metadata.get("resource_slots", {})
    if not isinstance(raw_slots, Mapping):
        raise SemanticError("resource_slots must be a mapping")
    queues: Dict[object, list] = {}
    for transfer in schedule.transfers:
        lane = LaneKey(
            transfer.src_rank,
            transfer.dst_rank,
            transfer.channel,
        )
        queues.setdefault(("lane", lane), []).append(transfer)
        slots = raw_slots.get(transfer.transfer_id, {})
        if not isinstance(slots, Mapping):
            raise SemanticError("resource slot assignment must be a mapping")
        for resource_id, slot in slots.items():
            queues.setdefault(
                ("resource", resource_id, slot),
                [],
            ).append(transfer)
    result = []
    for values in queues.values():
        values.sort(
            key=lambda item: (
                item.st_time,
                item.ed_time,
                item.transfer_id,
            )
        )
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
            values = transfer.predecessor_ids | frozenset(
                raw.get(transfer.transfer_id, ())
            )
        except TypeError as error:
            raise SemanticError(
                "semantic predecessor IDs must be iterable"
            ) from error
        if not values <= known:
            raise SemanticError("semantic predecessor is missing")
        result[transfer.transfer_id] = values
    return result


def _add_precedence(
    model,
    predecessor_id: str,
    successor_id: str,
    starts: Mapping[str, object],
    ends: Mapping[str, object],
    by_id: Mapping[str, object],
    name: str,
) -> None:
    predecessor_is_modeled = predecessor_id in ends
    successor_is_modeled = successor_id in starts
    if predecessor_is_modeled and successor_is_modeled:
        model.addConstr(
            starts[successor_id] >= ends[predecessor_id],
            name=name,
        )
    elif predecessor_is_modeled:
        model.addConstr(
            ends[predecessor_id] <= by_id[successor_id].st_time,
            name=name,
        )
    elif successor_is_modeled:
        model.addConstr(
            starts[successor_id] >= by_id[predecessor_id].ed_time,
            name=name,
        )


def _add_local_timing_scope(model, gp, schedule: Schedule, impact: ImpactClosure):
    by_id = {
        transfer.transfer_id: transfer for transfer in schedule.transfers
    }
    unknown = impact.transfer_ids - set(by_id)
    if unknown:
        raise SemanticError("impact closure contains an unknown transfer")
    modeled_ids = tuple(sorted(impact.transfer_ids))
    starts = {
        transfer_id: model.addVar(
            lb=0.0,
            vtype=gp.GRB.CONTINUOUS,
            name="start_{:04d}".format(index),
        )
        for index, transfer_id in enumerate(modeled_ids)
    }
    ends = {
        transfer_id: model.addVar(
            lb=0.0,
            vtype=gp.GRB.CONTINUOUS,
            name="end_{:04d}".format(index),
        )
        for index, transfer_id in enumerate(modeled_ids)
    }
    for index, transfer_id in enumerate(modeled_ids):
        transfer = by_id[transfer_id]
        model.addConstr(
            ends[transfer_id] - starts[transfer_id]
            == transfer.ed_time - transfer.st_time,
            name="duration_{:04d}".format(index),
        )
    constraint_index = 0
    for successor_id, predecessor_ids in _semantic_dependencies(
        schedule
    ).items():
        for predecessor_id in sorted(predecessor_ids):
            if (
                predecessor_id not in impact.transfer_ids
                and successor_id not in impact.transfer_ids
            ):
                continue
            _add_precedence(
                model,
                predecessor_id,
                successor_id,
                starts,
                ends,
                by_id,
                "dependency_{:06d}".format(constraint_index),
            )
            constraint_index += 1
    for queue in _ordered_queues(schedule):
        for predecessor_id, successor_id in zip(queue, queue[1:]):
            if (
                predecessor_id not in impact.transfer_ids
                and successor_id not in impact.transfer_ids
            ):
                continue
            _add_precedence(
                model,
                predecessor_id,
                successor_id,
                starts,
                ends,
                by_id,
                "resource_order_{:06d}".format(constraint_index),
            )
            constraint_index += 1
    return modeled_ids, starts, ends


def solve_local_repair(
    schedule: Schedule,
    hint: FlowReplacementHint,
    impact: ImpactClosure,
    overlay: TuningOverlay,
    topology: Topology,
    inputs: ResolvedInput,
    budget: ModelBudget,
) -> RepairResult:
    if not isinstance(schedule, Schedule):
        raise SemanticError("schedule must be a Schedule")
    if not isinstance(hint, FlowReplacementHint):
        raise SemanticError("hint must be a FlowReplacementHint")
    if not isinstance(impact, ImpactClosure):
        raise SemanticError("impact must be an ImpactClosure")
    if not isinstance(overlay, TuningOverlay):
        raise SemanticError("overlay must be a TuningOverlay")
    if not isinstance(topology, Topology):
        raise SemanticError("topology must be a Topology")
    if not isinstance(inputs, ResolvedInput):
        raise SemanticError("inputs must be a ResolvedInput")
    if not isinstance(budget, ModelBudget):
        raise SemanticError("budget must be a ModelBudget")
    known_transfer_ids = {
        transfer.transfer_id for transfer in schedule.transfers
    }
    if not impact.transfer_ids <= known_transfer_ids:
        raise SemanticError("impact closure contains an unknown transfer")
    if budget.seconds <= 0.0:
        return _result(
            RepairStatus.TIMEOUT,
            "local_repair_budget_exhausted",
            "local repair budget is exhausted",
            affected_transfer_count=len(impact.transfer_ids),
        )
    if not hint.candidate_flow_ids:
        return _result(
            RepairStatus.INFEASIBLE,
            "local_repair_has_no_candidate",
            "local repair has no candidate flow",
            affected_transfer_count=len(impact.transfer_ids),
        )
    try:
        overlay.validate_against(inputs, schedule, topology)
        _semantic_dependencies(schedule)
        _ordered_queues(schedule)
        flow = build_flow_index(schedule).flow(hint.source_flow_id)
        if flow.demand_id != hint.demand_id:
            raise SemanticError("hint demand does not match source flow")
        if hint.divergence_rank not in flow.ranks:
            raise SemanticError("hint divergence rank is outside the flow")
        divergence_index = flow.ranks.index(hint.divergence_rank)
        if (
            divergence_index >= flow.comparison_end
            or flow.transfer_ids[divergence_index]
            != hint.waiting_transfer_id
        ):
            raise SemanticError(
                "hint waiting transfer does not match source flow"
            )
        legal_options = _legal_options(
            flow,
            hint,
            overlay,
            topology,
            inputs,
            schedule,
        )
    except SemanticError as error:
        return _result(
            RepairStatus.INVALID,
            "local_repair_invalid",
            str(error),
            affected_transfer_count=len(impact.transfer_ids),
        )
    if not legal_options:
        return _result(
            RepairStatus.INFEASIBLE,
            "local_repair_has_no_legal_candidate",
            "local repair has no legal candidate flow",
            affected_transfer_count=len(impact.transfer_ids),
            legal_candidate_count=0,
        )
    legal_candidate_ids = tuple(
        option.candidate_id for option in legal_options
    )
    try:
        gp = GurobiAdapter.require()
        model = gp.Model("vericcl-local-repair")
        model.Params.OutputFlag = 0
        for name, value in overlay.milp_parameters:
            if name not in {"TimeLimit", "Seed"}:
                model.setParam(name, value)
        model.Params.TimeLimit = budget.seconds
        model.Params.Seed = inputs.solver.solver_seed
        modeled_ids, starts, ends = _add_local_timing_scope(
            model,
            gp,
            schedule,
            impact,
        )
        variables = {
            candidate_id: model.addVar(
                vtype=gp.GRB.BINARY,
                name="candidate_{:04d}".format(index),
            )
            for index, candidate_id in enumerate(legal_candidate_ids)
        }
        model.addConstr(
            gp.quicksum(variables.values()) == 1,
            name="select_one_suffix",
        )
        model.setObjective(
            gp.quicksum(
                index * variables[candidate_id]
                for index, candidate_id in enumerate(legal_candidate_ids)
            ),
            gp.GRB.MINIMIZE,
        )
        model.optimize()
    except Exception as error:
        return _result(
            RepairStatus.NOT_RUN,
            "local_repair_solver_unavailable",
            "local repair solver is unavailable",
            error_type=type(error).__name__,
            error=str(error),
            affected_transfer_count=len(impact.transfer_ids),
            modeled_transfer_ids=tuple(sorted(impact.transfer_ids)),
            legal_candidate_count=len(legal_candidate_ids),
        )

    if model.Status == gp.GRB.INFEASIBLE:
        return _result(
            RepairStatus.INFEASIBLE,
            "local_repair_infeasible",
            "local repair model is infeasible",
            affected_transfer_count=len(impact.transfer_ids),
            model_variable_count=len(variables) + len(starts) + len(ends),
        )
    if model.SolCount == 0:
        status = (
            RepairStatus.TIMEOUT
            if model.Status == gp.GRB.TIME_LIMIT
            else RepairStatus.INVALID
        )
        return _result(
            status,
            "local_repair_has_no_solution",
            "local repair model produced no solution",
            affected_transfer_count=len(impact.transfer_ids),
            model_variable_count=len(variables) + len(starts) + len(ends),
        )
    selected = min(
        (
            candidate_id
            for candidate_id, variable in variables.items()
            if variable.X > 0.5
        ),
        key=lambda value: hint.candidate_flow_ids.index(value),
    )
    narrowed = replace(
        hint,
        candidate_flow_ids=(selected,),
        candidate_paths={selected: hint.candidate_paths[selected]},
        candidate_first_lanes={
            selected: hint.candidate_first_lanes[selected]
        },
    )
    repaired = repair_flow_suffix(
        schedule,
        narrowed,
        overlay,
        topology,
        inputs,
    )
    evidence = dict(repaired.evidence)
    evidence.update(
        {
            "scope": "local",
            "affected_transfer_count": len(impact.transfer_ids),
            "modeled_transfer_ids": modeled_ids,
            "fixed_transfer_count": len(schedule.transfers) - len(modeled_ids),
            "timing_variable_count": len(starts) + len(ends),
            "model_variable_count": len(variables) + len(starts) + len(ends),
            "solver_status": int(model.Status),
            "legal_candidate_count": len(legal_candidate_ids),
        }
    )
    return replace(repaired, method="local_milp", evidence=evidence)
