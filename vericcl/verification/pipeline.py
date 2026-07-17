from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from vericcl.errors import SemanticError
from vericcl.input.models import ResolvedInput
from vericcl.semantics.atom import Schedule
from vericcl.topology.model import Topology
from vericcl.verification.bdd_backend import BDDAnalysisResult
from vericcl.verification.bdd_flow import analyze_flow_congestion
from vericcl.verification.bdd_order import analyze_tb_order
from vericcl.verification.constraints import verify_schedule_pre_lowering
from vericcl.verification.model import (
    CheckResult,
    ValidationReport,
    ValidationStatus,
)
from vericcl.verification.simulator import SimulationResult, simulate_schedule
from vericcl.xml.buffers import build_buffer_plan
from vericcl.xml.compatibility import check_msccl_compatibility
from vericcl.xml.compatibility import renumber_dependent_threadblocks
from vericcl.xml.deadlock import simulate_endpoint_execution
from vericcl.xml.dependencies import build_transfer_dag
from vericcl.xml.emitter import emit_xml
from vericcl.xml.endpoints import lower_endpoints
from vericcl.xml.liveness import verify_buffer_liveness
from vericcl.xml.list_scheduler import schedule_threadblocks
from vericcl.xml.lower import XmlArtifact
from vericcl.xml.parser import validate_xml


_PRE_DIMENSIONS = (
    "semantic",
    "state",
    "endpoint",
    "topology",
    "timing",
    "resource",
)


@dataclass(frozen=True)
class VerificationOutcome:
    report: ValidationReport
    artifact: Optional[XmlArtifact]
    simulation: Optional[SimulationResult]
    flow_bdd: Optional[BDDAnalysisResult]
    order_bdd: Optional[BDDAnalysisResult]

    def __post_init__(self) -> None:
        if not isinstance(self.report, ValidationReport):
            raise SemanticError("verification outcome report is invalid")
        if self.artifact is not None and not isinstance(
            self.artifact,
            XmlArtifact,
        ):
            raise SemanticError("verification outcome artifact is invalid")
        if self.simulation is not None and not isinstance(
            self.simulation,
            SimulationResult,
        ):
            raise SemanticError("verification outcome simulation is invalid")
        for value in (self.flow_bdd, self.order_bdd):
            if value is not None and not isinstance(value, BDDAnalysisResult):
                raise SemanticError("verification outcome BDD result is invalid")


def _check(
    dimension: str,
    status: ValidationStatus,
    code: str,
    message: str,
    evidence: Optional[Mapping[str, object]] = None,
) -> CheckResult:
    return CheckResult(
        dimension=dimension,
        status=status,
        code=code,
        message=message,
        evidence={} if evidence is None else evidence,
    )


def _not_run(dimension: str, prerequisite: str) -> CheckResult:
    return _check(
        dimension,
        ValidationStatus.NOT_RUN,
        "{}_prerequisite_not_met".format(dimension),
        "{} validation did not run".format(dimension),
        {"prerequisite": prerequisite},
    )


def _input_result(inputs: ResolvedInput) -> CheckResult:
    return _check(
        "input",
        ValidationStatus.VALID,
        "resolved_input_valid",
        "resolved input validation passed",
        {"normalized_input_sha256": inputs.input_sha256},
    )


def _pre_results(
    schedule: Schedule,
    inputs: ResolvedInput,
    topology: Topology,
) -> Mapping[str, CheckResult]:
    values = verify_schedule_pre_lowering(schedule, inputs, topology)
    result = {item.dimension: item for item in values}
    if set(result) != set(_PRE_DIMENSIONS):
        raise SemanticError("pre-lowering validation dimensions are incomplete")
    return result


def _base_values(
    inputs: ResolvedInput,
    pre: Mapping[str, CheckResult],
) -> dict:
    values = {
        "input": _input_result(inputs),
        **pre,
    }
    for dimension in (
        "buffer",
        "deadlock",
        "xml",
        "bdd",
        "simulation",
        "runtime",
        "online",
    ):
        values[dimension] = _not_run(dimension, "pre_lowering")
    return values


def _pre_lowering_valid(pre: Mapping[str, CheckResult]) -> bool:
    return all(
        result.status is ValidationStatus.VALID
        for result in pre.values()
    )


def _outcome(
    values: Mapping[str, CheckResult],
    artifact: Optional[XmlArtifact] = None,
    simulation: Optional[SimulationResult] = None,
    flow_bdd: Optional[BDDAnalysisResult] = None,
    order_bdd: Optional[BDDAnalysisResult] = None,
) -> VerificationOutcome:
    return VerificationOutcome(
        report=ValidationReport(**dict(values)),
        artifact=artifact,
        simulation=simulation,
        flow_bdd=flow_bdd,
        order_bdd=order_bdd,
    )


def _buffer_summary(artifact: XmlArtifact) -> Mapping[str, object]:
    plan = artifact.buffer_plan
    return {
        "slice_count": plan.slice_count,
        "local_copy_count": len(plan.local_copies),
        "input_chunks": sum(plan.i_chunks.values()),
        "output_chunks": sum(plan.o_chunks.values()),
        "scratch_chunks": sum(plan.s_chunks.values()),
    }


def _buffer_plan_summary(plan) -> Mapping[str, object]:
    return {
        "slice_count": plan.slice_count,
        "local_copy_count": len(plan.local_copies),
        "input_chunks": sum(plan.i_chunks.values()),
        "output_chunks": sum(plan.o_chunks.values()),
        "scratch_chunks": sum(plan.s_chunks.values()),
    }


def _bdd_result(
    flow: BDDAnalysisResult,
    order: BDDAnalysisResult,
) -> CheckResult:
    status = (
        ValidationStatus.VALID
        if flow.status is ValidationStatus.VALID
        and order.status is ValidationStatus.VALID
        else ValidationStatus.ANALYSIS_ERROR
    )
    return _check(
        "bdd",
        status,
        (
            "bdd_analysis_complete"
            if status is ValidationStatus.VALID
            else "bdd_analysis_error"
        ),
        (
            "BDD opportunity analysis completed"
            if status is ValidationStatus.VALID
            else "BDD opportunity analysis failed"
        ),
        {
            "flow": {
                "status": flow.status.value,
                "code": flow.code,
                "hint_count": len(flow.hints),
                "evidence": dict(flow.evidence),
            },
            "order": {
                "status": order.status.value,
                "code": order.code,
                "hint_count": len(order.hints),
                "evidence": dict(order.evidence),
            },
        },
    )


def _complete_analysis(
    schedule: Schedule,
    artifact: XmlArtifact,
    inputs: ResolvedInput,
    topology: Topology,
    values: Mapping[str, CheckResult],
) -> VerificationOutcome:
    results = dict(values)
    compatibility = check_msccl_compatibility(artifact)
    artifact = compatibility.apply(artifact)
    results["runtime"] = _check(
        "runtime",
        (
            ValidationStatus.VALID
            if compatibility.runtime_compatible
            else ValidationStatus.WARNING
        ),
        (
            "msccl_runtime_compatible"
            if compatibility.runtime_compatible
            else "msccl_runtime_incompatible"
        ),
        (
            "MSCCL execution compatibility passed"
            if compatibility.runtime_compatible
            else "MSCCL execution compatibility limits were exceeded"
        ),
        {
            "issues": tuple(
                {
                    "code": issue.code,
                    "rank": issue.rank,
                    "tb_id": issue.tb_id,
                    "channel": issue.channel,
                    "current_value": issue.current_value,
                    "limit": issue.limit,
                    "transfer_ids": issue.transfer_ids,
                }
                for issue in compatibility.issues
            ),
        },
    )

    flow_bdd = analyze_flow_congestion(schedule, topology, inputs)
    order_bdd = analyze_tb_order(artifact.tb_program, schedule)
    results["bdd"] = _bdd_result(flow_bdd, order_bdd)

    try:
        simulation = simulate_schedule(schedule, topology)
    except SemanticError as error:
        results["simulation"] = _check(
            "simulation",
            ValidationStatus.FAILED,
            "dynamic_simulation_failed",
            str(error),
        )
        simulation = None
    else:
        results["simulation"] = _check(
            "simulation",
            ValidationStatus.VALID,
            "dynamic_simulation_complete",
            "dynamic concurrency simulation completed",
            {
                "completion_time_us": simulation.completion_time_us,
                "event_count": len(simulation.events),
                "queue_wait_times": dict(simulation.queue_wait_times),
            },
        )

    results["online"] = _not_run(
        "online",
        "online_validation_not_requested",
    )
    return _outcome(
        results,
        artifact,
        simulation,
        flow_bdd,
        order_bdd,
    )


def _verify_candidate_outcome(
    schedule: Schedule,
    artifact: XmlArtifact,
    inputs: ResolvedInput,
    topology: Topology,
) -> VerificationOutcome:
    if not isinstance(schedule, Schedule):
        raise SemanticError("schedule must be a Schedule")
    if not isinstance(artifact, XmlArtifact):
        raise SemanticError("artifact must be an XmlArtifact")
    if not isinstance(inputs, ResolvedInput):
        raise SemanticError("inputs must be a ResolvedInput")
    if not isinstance(topology, Topology):
        raise SemanticError("topology must be a Topology")
    pre = _pre_results(schedule, inputs, topology)
    values = _base_values(inputs, pre)
    if not _pre_lowering_valid(pre):
        return _outcome(values, artifact)

    try:
        verify_buffer_liveness(schedule, artifact.buffer_plan, inputs)
    except SemanticError as error:
        values["buffer"] = _check(
            "buffer",
            ValidationStatus.INVALID,
            "buffer_liveness_invalid",
            str(error),
        )
        return _outcome(values, artifact)
    values["buffer"] = _check(
        "buffer",
        ValidationStatus.VALID,
        "buffer_liveness_valid",
        "buffer liveness validation passed",
        _buffer_summary(artifact),
    )

    expected_transfer_ids = {
        transfer.transfer_id for transfer in schedule.transfers
    }
    actual_transfer_ids = set(artifact.endpoint_program.by_transfer_id)
    if actual_transfer_ids != expected_transfer_ids:
        values["endpoint"] = _check(
            "endpoint",
            ValidationStatus.INVALID,
            "endpoint_schedule_binding_mismatch",
            "endpoint program does not match schedule transfers",
            {
                "expected_transfer_ids": tuple(sorted(expected_transfer_ids)),
                "actual_transfer_ids": tuple(sorted(actual_transfer_ids)),
            },
        )
        return _outcome(values, artifact)
    values["endpoint"] = _check(
        "endpoint",
        ValidationStatus.VALID,
        "endpoint_program_valid",
        "endpoint program validation passed",
        {"endpoint_count": len(artifact.endpoint_program.endpoints)},
    )

    deadlock = simulate_endpoint_execution(artifact.tb_program)
    if deadlock.deadlocked:
        values["deadlock"] = _check(
            "deadlock",
            ValidationStatus.INVALID,
            "threadblock_deadlock",
            "threadblock program is deadlocked",
            {
                "blocked_transfer_ids": tuple(
                    sorted(deadlock.blocked_transfer_ids)
                ),
            },
        )
        return _outcome(values, artifact)
    values["deadlock"] = _check(
        "deadlock",
        ValidationStatus.VALID,
        "threadblock_deadlock_free",
        "threadblock deadlock validation passed",
        {"completed_step_count": len(deadlock.completed_step_ids)},
    )

    try:
        validate_xml(
            artifact.xml_text,
            artifact.tb_program,
            artifact.buffer_plan,
            inputs,
        )
    except SemanticError as error:
        values["xml"] = _check(
            "xml",
            ValidationStatus.INVALID,
            "xml_validation_failed",
            str(error),
        )
        return _outcome(values, artifact)
    values["xml"] = _check(
        "xml",
        ValidationStatus.VALID,
        "xml_validation_passed",
        "XML validation passed",
        {"sha256": artifact.sha256},
    )
    return _complete_analysis(
        schedule,
        artifact,
        inputs,
        topology,
        values,
    )


def verify_candidate(
    schedule: Schedule,
    artifact: XmlArtifact,
    inputs: ResolvedInput,
    topology: Topology,
) -> ValidationReport:
    return _verify_candidate_outcome(
        schedule,
        artifact,
        inputs,
        topology,
    ).report


def verify_candidate_outcome(
    schedule: Schedule,
    artifact: XmlArtifact,
    inputs: ResolvedInput,
    topology: Topology,
) -> VerificationOutcome:
    return _verify_candidate_outcome(
        schedule,
        artifact,
        inputs,
        topology,
    )


def validate_and_lower_candidate(
    schedule: Schedule,
    inputs: ResolvedInput,
    topology: Topology,
) -> VerificationOutcome:
    if not isinstance(schedule, Schedule):
        raise SemanticError("schedule must be a Schedule")
    if not isinstance(inputs, ResolvedInput):
        raise SemanticError("inputs must be a ResolvedInput")
    if not isinstance(topology, Topology):
        raise SemanticError("topology must be a Topology")
    pre = _pre_results(schedule, inputs, topology)
    values = _base_values(inputs, pre)
    if not _pre_lowering_valid(pre):
        return _outcome(values)

    try:
        buffers = build_buffer_plan(schedule, inputs)
        verify_buffer_liveness(schedule, buffers, inputs)
    except SemanticError as error:
        values["buffer"] = _check(
            "buffer",
            ValidationStatus.INVALID,
            "buffer_plan_failed",
            str(error),
        )
        return _outcome(values)
    values["buffer"] = _check(
        "buffer",
        ValidationStatus.VALID,
        "buffer_liveness_valid",
        "buffer liveness validation passed",
        _buffer_plan_summary(buffers),
    )

    try:
        endpoints = lower_endpoints(schedule, buffers)
        dag = build_transfer_dag(endpoints, schedule, buffers)
    except SemanticError as error:
        values["endpoint"] = _check(
            "endpoint",
            ValidationStatus.INVALID,
            "endpoint_lowering_failed",
            str(error),
        )
        return _outcome(values)
    values["endpoint"] = _check(
        "endpoint",
        ValidationStatus.VALID,
        "endpoint_program_valid",
        "endpoint program validation passed",
        {"endpoint_count": len(endpoints.endpoints)},
    )

    try:
        threadblocks = schedule_threadblocks(endpoints, dag)
        threadblocks = renumber_dependent_threadblocks(threadblocks)
    except SemanticError as error:
        values["deadlock"] = _check(
            "deadlock",
            ValidationStatus.INVALID,
            "threadblock_scheduling_failed",
            str(error),
        )
        return _outcome(values)

    deadlock = simulate_endpoint_execution(threadblocks)
    if deadlock.deadlocked:
        values["deadlock"] = _check(
            "deadlock",
            ValidationStatus.INVALID,
            "threadblock_deadlock",
            "threadblock program is deadlocked",
            {
                "blocked_transfer_ids": tuple(
                    sorted(deadlock.blocked_transfer_ids)
                ),
            },
        )
        return _outcome(values)
    values["deadlock"] = _check(
        "deadlock",
        ValidationStatus.VALID,
        "threadblock_deadlock_free",
        "threadblock deadlock validation passed",
        {"completed_step_count": len(deadlock.completed_step_ids)},
    )

    try:
        xml_text = emit_xml(threadblocks, buffers, inputs)
    except SemanticError as error:
        values["xml"] = _check(
            "xml",
            ValidationStatus.INVALID,
            "xml_emission_failed",
            str(error),
        )
        return _outcome(values)
    try:
        validate_xml(xml_text, threadblocks, buffers, inputs)
    except SemanticError as error:
        values["xml"] = _check(
            "xml",
            ValidationStatus.INVALID,
            "xml_validation_failed",
            str(error),
        )
        return _outcome(values)
    artifact = XmlArtifact(
        xml_text=xml_text,
        buffer_plan=buffers,
        endpoint_program=endpoints,
        tb_program=threadblocks,
        sha256=hashlib.sha256(xml_text.encode("utf-8")).hexdigest(),
        runtime_compatible=True,
    )
    values["xml"] = _check(
        "xml",
        ValidationStatus.VALID,
        "xml_validation_passed",
        "XML validation passed",
        {"sha256": artifact.sha256},
    )
    return _complete_analysis(
        schedule,
        artifact,
        inputs,
        topology,
        values,
    )
