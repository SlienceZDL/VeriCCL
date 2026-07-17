from dataclasses import replace

import pytest

from vericcl.input.models import ObjectiveMode
from vericcl.errors import SemanticError
from vericcl.solver.model import (
    SolveCandidate,
    SolveStatus,
    SolverMetrics,
)
from vericcl.tuning.engine import TuningContext, tune
from vericcl.verification.bdd_backend import BDDAnalysisResult
from vericcl.verification.model import ValidationStatus
from vericcl.verification.pipeline import validate_and_lower_candidate
from vericcl.verification.pipeline import verify_candidate
from vericcl.xml.deadlock import DeadlockResult

from tests.unit.verification.helpers import inputs, topology
from tests.unit.xml.helpers import two_rank_allreduce_schedule


pytestmark = pytest.mark.phase05


def test_complete_validation_pipeline_binds_schedule_xml_bdd_and_simulation():
    outcome = validate_and_lower_candidate(
        two_rank_allreduce_schedule(),
        inputs(),
        topology(),
    )

    assert outcome.artifact is not None
    assert outcome.report.overall_status is ValidationStatus.VALID
    assert outcome.report.eligible_for_selection is True
    assert outcome.report.xml.evidence["sha256"] == outcome.artifact.sha256
    assert outcome.report.bdd.status is ValidationStatus.VALID
    assert outcome.report.simulation.status is ValidationStatus.VALID
    assert outcome.simulation.completion_time_us > 0.0


def test_invalid_pre_lowering_schedule_never_builds_buffers_or_xml(monkeypatch):
    called = []

    def forbidden_lower(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError("lowering must not run")

    monkeypatch.setattr(
        "vericcl.verification.pipeline.build_buffer_plan",
        forbidden_lower,
    )
    outcome = validate_and_lower_candidate(
        replace(two_rank_allreduce_schedule(), rank_count=3),
        inputs(),
        topology(),
    )

    assert called == []
    assert outcome.artifact is None
    assert outcome.report.semantic.status is ValidationStatus.INVALID
    assert outcome.report.buffer.status is ValidationStatus.NOT_RUN
    assert outcome.report.xml.status is ValidationStatus.NOT_RUN
    assert outcome.report.bdd.status is ValidationStatus.NOT_RUN


def test_bdd_analysis_error_blocks_selection_without_hiding_correctness(
    monkeypatch,
):
    failure = BDDAnalysisResult(
        status=ValidationStatus.ANALYSIS_ERROR,
        code="forced_bdd_error",
        message="forced BDD analysis error",
        hints=(),
        evidence={"forced": True},
    )
    monkeypatch.setattr(
        "vericcl.verification.pipeline.analyze_flow_congestion",
        lambda *args: failure,
    )

    outcome = validate_and_lower_candidate(
        two_rank_allreduce_schedule(),
        inputs(),
        topology(),
    )

    assert outcome.report.semantic.status is ValidationStatus.VALID
    assert outcome.report.bdd.status is ValidationStatus.ANALYSIS_ERROR
    assert outcome.report.overall_status is ValidationStatus.VALID
    assert outcome.report.eligible_for_selection is False


def test_default_tuning_pipeline_keeps_fully_validated_best_candidate():
    schedule = two_rank_allreduce_schedule()
    candidate = SolveCandidate(
        candidate_id="integration-initial",
        node_schedules={"global": schedule},
        objective_mode=ObjectiveMode.LATENCY,
        channel_count=1,
        metrics=SolverMetrics(
            status=SolveStatus.FEASIBLE,
            objective_values=(2.0,),
            best_bound=0.0,
            mip_gap=0.0,
            within_requested_gap=False,
            solve_time_s=0.0,
            model_count=0,
            operation_count=2,
            hop_count=2,
            makespan_us=2.0,
            maximum_normalized_resource_load=2.0,
            solver_name="integration",
            solver_version="1",
            solver_seed=0,
            thread_count=1,
            termination_reason="integration_complete",
        ),
        selected_best=False,
        proven_optimal=False,
        search_space_restricted=False,
        restrictions=(),
        parent_candidate_id=None,
    )
    result = tune(
        candidate,
        TuningContext(
            inputs=inputs(),
            topology=topology(),
            initial_schedule=schedule,
            max_iterations=1,
            timeout_s=10.0,
        ),
    )

    assert result.selected_candidate_id == "integration-initial"
    assert result.selected_artifact is not None
    assert len(result.history) == 1
    assert result.history[0].report.eligible_for_selection is True
    assert result.stop_reason == "candidate_space_exhausted"


@pytest.mark.parametrize("failure", ("buffer", "deadlock", "xml"))
def test_post_lowering_failures_remain_dimension_specific(monkeypatch, failure):
    schedule = two_rank_allreduce_schedule()
    input_value = inputs()
    topology_value = topology()
    artifact = validate_and_lower_candidate(
        schedule,
        input_value,
        topology_value,
    ).artifact

    if failure == "buffer":
        def fail_buffer(*args):
            raise SemanticError("forced buffer failure")

        monkeypatch.setattr(
            "vericcl.verification.pipeline.verify_buffer_liveness",
            fail_buffer,
        )
    elif failure == "deadlock":
        monkeypatch.setattr(
            "vericcl.verification.pipeline.simulate_endpoint_execution",
            lambda program: DeadlockResult(
                True,
                frozenset(),
                frozenset({"allreduce-send"}),
                {(0, 0): "blocked"},
            ),
        )
    else:
        def fail_xml(*args):
            raise SemanticError("forced XML failure")

        monkeypatch.setattr(
            "vericcl.verification.pipeline.validate_xml",
            fail_xml,
        )

    report = verify_candidate(
        schedule,
        artifact,
        input_value,
        topology_value,
    )

    result = getattr(report, failure)
    assert result.status is ValidationStatus.INVALID
    assert report.eligible_for_selection is False


@pytest.mark.parametrize(
    ("failure", "dimension", "expected_code"),
    (
        ("buffer", "buffer", "buffer_plan_failed"),
        ("endpoint", "endpoint", "endpoint_lowering_failed"),
        (
            "threadblock",
            "deadlock",
            "threadblock_scheduling_failed",
        ),
        ("deadlock", "deadlock", "threadblock_deadlock"),
        ("xml_emission", "xml", "xml_emission_failed"),
        ("xml_validation", "xml", "xml_validation_failed"),
    ),
)
def test_lowering_failure_is_attributed_to_exact_stage(
    monkeypatch,
    failure,
    dimension,
    expected_code,
):
    def fail_stage(*args):
        raise SemanticError("forced {} failure".format(failure))

    if failure == "deadlock":
        monkeypatch.setattr(
            "vericcl.verification.pipeline.simulate_endpoint_execution",
            lambda program: DeadlockResult(
                True,
                frozenset(),
                frozenset({"allreduce-send"}),
                {(0, 0): "blocked"},
            ),
        )
    else:
        target = {
            "buffer": "build_buffer_plan",
            "endpoint": "lower_endpoints",
            "threadblock": "schedule_threadblocks",
            "xml_emission": "emit_xml",
            "xml_validation": "validate_xml",
        }[failure]
        monkeypatch.setattr(
            "vericcl.verification.pipeline.{}".format(target),
            fail_stage,
        )
    outcome = validate_and_lower_candidate(
        two_rank_allreduce_schedule(),
        inputs(),
        topology(),
    )

    assert outcome.artifact is None
    result = getattr(outcome.report, dimension)
    assert result.status is ValidationStatus.INVALID
    assert result.code == expected_code
    assert outcome.report.bdd.status is ValidationStatus.NOT_RUN


def test_endpoint_program_must_bind_the_exact_schedule(monkeypatch):
    schedule = two_rank_allreduce_schedule()
    input_value = inputs()
    topology_value = topology()
    artifact = validate_and_lower_candidate(
        schedule,
        input_value,
        topology_value,
    ).artifact
    renamed = "renamed-allreduce-send"
    transfers = tuple(
        replace(transfer, transfer_id=renamed)
        if transfer.transfer_id == "allreduce-send"
        else transfer
        for transfer in schedule.transfers
    )
    metadata = dict(schedule.metadata)
    semantic = dict(metadata["semantic_predecessors"])
    semantic[renamed] = semantic.pop("allreduce-send")
    metadata["semantic_predecessors"] = semantic
    dependencies = dict(metadata["final_dependencies"])
    metadata["final_dependencies"] = {
        key: tuple(
            renamed if value == "allreduce-send" else value
            for value in values
        )
        for key, values in dependencies.items()
    }
    renamed_schedule = replace(
        schedule,
        transfers=transfers,
        metadata=metadata,
    )
    monkeypatch.setattr(
        "vericcl.verification.pipeline.verify_buffer_liveness",
        lambda *args: None,
    )

    report = verify_candidate(
        renamed_schedule,
        artifact,
        input_value,
        topology_value,
    )

    assert report.endpoint.status is ValidationStatus.INVALID
    assert report.endpoint.code == "endpoint_schedule_binding_mismatch"


def test_dynamic_simulation_failure_is_not_reported_as_correctness_success(
    monkeypatch,
):
    schedule = two_rank_allreduce_schedule()
    input_value = inputs()
    topology_value = topology()
    artifact = validate_and_lower_candidate(
        schedule,
        input_value,
        topology_value,
    ).artifact

    def fail_simulation(*args):
        raise SemanticError("forced simulation failure")

    monkeypatch.setattr(
        "vericcl.verification.pipeline.simulate_schedule",
        fail_simulation,
    )
    report = verify_candidate(
        schedule,
        artifact,
        input_value,
        topology_value,
    )

    assert report.simulation.status is ValidationStatus.FAILED
    assert report.overall_status is ValidationStatus.FAILED
    assert report.eligible_for_selection is False


def test_validation_pipeline_runs_lowering_and_analysis_once_in_order(
    monkeypatch,
):
    order = []
    targets = (
        "build_buffer_plan",
        "verify_buffer_liveness",
        "lower_endpoints",
        "build_transfer_dag",
        "schedule_threadblocks",
        "simulate_endpoint_execution",
        "emit_xml",
        "validate_xml",
        "check_msccl_compatibility",
        "analyze_flow_congestion",
        "analyze_tb_order",
        "simulate_schedule",
    )

    for target in targets:
        module = __import__(
            "vericcl.verification.pipeline",
            fromlist=[target],
        )
        original = getattr(module, target)

        def wrapped(*args, _target=target, _original=original, **kwargs):
            order.append(_target)
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, target, wrapped)

    outcome = validate_and_lower_candidate(
        two_rank_allreduce_schedule(),
        inputs(),
        topology(),
    )

    assert outcome.report.overall_status is ValidationStatus.VALID
    assert order == list(targets)
