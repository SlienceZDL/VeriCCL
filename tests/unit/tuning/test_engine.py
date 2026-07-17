from dataclasses import replace

import pytest

from vericcl.errors import SemanticError
from vericcl.input.models import ObjectiveMode
from vericcl.solver.model import (
    SolveCandidate,
    SolveStatus,
    SolverMetrics,
)
from vericcl.tuning.engine import (
    _repair_tb_order,
    CandidateAssessment,
    CandidateProposal,
    OnlinePerformance,
    TuningContext,
    TuningResult,
    tune,
)
from vericcl.tuning.model import TuningOverlay
from vericcl.verification.bdd_flow import analyze_flow_congestion
from vericcl.verification.bdd_order import analyze_tb_order
from vericcl.verification.model import (
    CheckResult,
    ValidationReport,
    ValidationStatus,
)
from vericcl.verification.pipeline import VerificationOutcome

from tests.unit.tuning.helpers import waiting_case
from tests.unit.verification.bdd_helpers import tb_order_case
from tests.unit.verification.helpers import inputs, topology
from tests.unit.xml.helpers import two_rank_allreduce_schedule


pytestmark = pytest.mark.phase05


_DIMENSIONS = (
    "input",
    "semantic",
    "state",
    "topology",
    "timing",
    "resource",
    "buffer",
    "endpoint",
    "deadlock",
    "xml",
    "bdd",
    "simulation",
    "runtime",
    "online",
)


def _report(**statuses):
    return ValidationReport(
        **{
            dimension: CheckResult(
                dimension,
                statuses.get(dimension, ValidationStatus.VALID),
                "{}_status".format(dimension),
                "{} check completed".format(dimension),
                {},
            )
            for dimension in _DIMENSIONS
        }
    )


def _candidate():
    schedule = two_rank_allreduce_schedule()
    return SolveCandidate(
        candidate_id="initial",
        node_schedules={"global": schedule},
        objective_mode=ObjectiveMode.LATENCY,
        channel_count=1,
        metrics=SolverMetrics(
            status=SolveStatus.FEASIBLE,
            objective_values=(10.0,),
            best_bound=0.0,
            mip_gap=0.0,
            within_requested_gap=False,
            solve_time_s=0.0,
            model_count=0,
            operation_count=2,
            hop_count=2,
            makespan_us=10.0,
            maximum_normalized_resource_load=2.0,
            solver_name="test",
            solver_version="1",
            solver_seed=0,
            thread_count=1,
            termination_reason="test",
        ),
        selected_best=False,
        proven_optimal=False,
        search_space_restricted=False,
        restrictions=(),
        parent_candidate_id=None,
    )


def _proposal(candidate_id):
    return CandidateProposal(
        candidate_id=candidate_id,
        schedule=replace(
            two_rank_allreduce_schedule(),
            schedule_id=candidate_id,
        ),
        overlay=TuningOverlay(
            overlay_id="{}-overlay".format(candidate_id),
            parent_candidate_id="initial",
            path_weights=((candidate_id, 1.0),),
        ),
        parent_candidate_id="initial",
        tuning_strategy={"kind": "test"},
    )


def _context(assessments, proposals, *, online=False):
    def assess(proposal):
        return assessments[proposal.candidate_id]

    def generate(_, iteration):
        return tuple(proposals) if iteration == 0 else ()

    return TuningContext(
        inputs=inputs(),
        topology=topology(),
        initial_schedule=two_rank_allreduce_schedule(),
        plan=None,
        assess=assess,
        generate=generate,
        simulate=lambda schedule: assessments[
            schedule.schedule_id
        ].simulation_time_us,
        online_validation=online,
        max_iterations=1,
        timeout_s=100.0,
        clock=lambda: 0.0,
    )


def test_offline_tuning_uses_strict_improvement_and_best_history():
    initial = CandidateAssessment(_report(), None, 10.0, None)
    fast = CandidateAssessment(_report(), None, 8.0, None)
    slower = CandidateAssessment(_report(), None, 9.0, None)
    equal = CandidateAssessment(_report(), None, 10.0, None)
    assessments = {
        "initial": initial,
        "candidate-fast": fast,
        "candidate-slower": slower,
        "candidate-equal": equal,
    }
    proposals = tuple(
        _proposal(value)
        for value in (
            "candidate-fast",
            "candidate-equal",
            "candidate-slower",
        )
    )

    result = tune(
        _candidate(),
        _context(assessments, proposals),
    )
    by_id = {entry.candidate_id: entry for entry in result.history}

    assert result.selected_candidate_id == "candidate-fast"
    assert by_id["candidate-fast"].accepted is True
    assert by_id["candidate-fast"].selected_best is True
    assert by_id["candidate-slower"].rejection_reason == (
        "no_simulated_improvement"
    )
    assert by_id["candidate-equal"].rejection_reason == (
        "no_simulated_improvement"
    )
    assert result.history[-1].candidate_id == "candidate-slower"
    assert result.history[-1].selected_best is False


def test_online_tuning_uses_median_and_cv_threshold():
    valid = _report()
    assessments = {
        "initial": CandidateAssessment(
            valid,
            None,
            10.0,
            OnlinePerformance(100.0, 0.01),
        ),
        "candidate-noisy": CandidateAssessment(
            valid,
            None,
            8.0,
            OnlinePerformance(98.0, 0.02),
        ),
        "candidate-stable": CandidateAssessment(
            valid,
            None,
            9.0,
            OnlinePerformance(95.0, 0.01),
        ),
    }
    proposals = (
        _proposal("candidate-noisy"),
        _proposal("candidate-stable"),
    )

    result = tune(
        _candidate(),
        _context(assessments, proposals, online=True),
    )
    by_id = {entry.candidate_id: entry for entry in result.history}

    assert by_id["candidate-noisy"].rejection_reason == (
        "online_improvement_below_threshold"
    )
    assert by_id["candidate-noisy"].required_improvement == pytest.approx(0.04)
    assert by_id["candidate-stable"].accepted is True
    assert result.selected_candidate_id == "candidate-stable"


def test_invalid_bdd_runtime_and_duplicate_candidates_are_retained():
    invalid = _report(state=ValidationStatus.INVALID)
    bdd_error = _report(bdd=ValidationStatus.ANALYSIS_ERROR)
    runtime_warning = _report(runtime=ValidationStatus.WARNING)
    valid = _report()
    duplicate_schedule = replace(
        two_rank_allreduce_schedule(),
        schedule_id="candidate-duplicate-one",
    )
    duplicate = CandidateProposal(
        candidate_id="candidate-duplicate-two",
        schedule=duplicate_schedule,
        overlay=TuningOverlay(
            overlay_id="duplicate-overlay-two",
            parent_candidate_id="initial",
            path_weights=(("duplicate-path", 1.0),),
        ),
        parent_candidate_id="initial",
        tuning_strategy={"kind": "test"},
    )
    first_duplicate = replace(
        duplicate,
        candidate_id="candidate-duplicate-one",
        overlay=replace(
            duplicate.overlay,
            overlay_id="duplicate-overlay-one",
        ),
    )
    assessments = {
        "initial": CandidateAssessment(valid, None, 10.0, None),
        "candidate-invalid": CandidateAssessment(invalid, None, 1.0, None),
        "candidate-bdd-error": CandidateAssessment(
            bdd_error,
            None,
            1.0,
            None,
        ),
        "candidate-runtime": CandidateAssessment(
            runtime_warning,
            None,
            1.0,
            None,
        ),
        "candidate-duplicate-one": CandidateAssessment(
            valid,
            None,
            9.0,
            None,
        ),
    }
    proposals = (
        _proposal("candidate-invalid"),
        _proposal("candidate-bdd-error"),
        _proposal("candidate-runtime"),
        first_duplicate,
        duplicate,
    )

    result = tune(
        _candidate(),
        _context(assessments, proposals),
    )
    by_id = {entry.candidate_id: entry for entry in result.history}

    assert by_id["candidate-invalid"].rejection_reason == "correctness_invalid"
    assert by_id["candidate-bdd-error"].rejection_reason == "bdd_analysis_error"
    assert by_id["candidate-runtime"].rejection_reason == "runtime_incompatible"
    assert by_id["candidate-runtime"].offline_analysis_only is True
    assert by_id["candidate-duplicate-two"].rejection_reason == (
        "duplicate_candidate_signature"
    )
    assert "candidate-runtime" in {
        entry.candidate_id for entry in result.history
    }


def test_incremental_simulation_rejects_before_complete_validation():
    assessment_calls = []
    assessments = {
        "initial": CandidateAssessment(_report(), None, 10.0, None),
        "candidate-slower": CandidateAssessment(_report(), None, 12.0, None),
    }

    def assess(proposal):
        assessment_calls.append(proposal.candidate_id)
        return assessments[proposal.candidate_id]

    context = replace(
        _context(assessments, (_proposal("candidate-slower"),)),
        assess=assess,
        simulate=lambda schedule: (
            10.0 if schedule.schedule_id != "candidate-slower" else 12.0
        ),
    )

    result = tune(_candidate(), context)
    by_id = {entry.candidate_id: entry for entry in result.history}

    assert assessment_calls == ["initial"]
    assert by_id["candidate-slower"].rejection_reason == (
        "no_simulated_improvement"
    )
    assert by_id["candidate-slower"].report is None


def test_builtin_generator_consumes_flow_bdd_hint_without_mutating_parent():
    schedule, topology_value, input_value, _ = waiting_case()
    flow_bdd = analyze_flow_congestion(schedule, topology_value, input_value)
    outcome = VerificationOutcome(
        _report(),
        None,
        None,
        flow_bdd,
        None,
    )
    initial_assessment = CandidateAssessment(
        _report(),
        None,
        10.0,
        None,
        outcome,
    )
    parent_snapshot = repr(schedule)

    def assess(proposal):
        if proposal.candidate_id == "initial":
            return initial_assessment
        return CandidateAssessment(_report(), None, 8.0, None)

    context = TuningContext(
        inputs=input_value,
        topology=topology_value,
        initial_schedule=schedule,
        assess=assess,
        generate=None,
        simulate=lambda value: 10.0 if value is schedule else 8.0,
        max_iterations=1,
        timeout_s=100.0,
        clock=lambda: 0.0,
    )

    result = tune(_candidate(), context)

    assert result.selected_candidate_id != "initial"
    selected = next(entry for entry in result.history if entry.selected_best)
    assert selected.tuning_strategy["kind"] == "flow_suffix"
    assert selected.parent_candidate_id == "initial"
    assert repr(schedule) == parent_snapshot


def test_tb_order_hint_reorders_only_nonsemantic_lane_precedence():
    program, schedule = tb_order_case()
    hint = analyze_tb_order(program, schedule).hints[0]

    repaired = _repair_tb_order(schedule, program, hint, "order-overlay")
    lane = tuple(
        transfer.transfer_id
        for transfer in sorted(
            (
                transfer
                for transfer in repaired.transfers
                if (transfer.src_rank, transfer.dst_rank, transfer.channel)
                == (0, 1, 0)
            ),
            key=lambda transfer: (
                transfer.st_time,
                transfer.ed_time,
                transfer.transfer_id,
            ),
        )
    )

    assert lane == ("fast", "slow")
    assert repaired.metadata["semantic_predecessors"] == (
        schedule.metadata["semantic_predecessors"]
    )
    assert "fast" in next(
        transfer.predecessor_ids
        for transfer in repaired.transfers
        if transfer.transfer_id == "slow"
    )


def test_tuning_is_bounded_by_twenty_iterations_and_wall_clock():
    valid = _report()

    def assess(proposal):
        if proposal.candidate_id == "initial":
            value = 100.0
        else:
            value = 99.0 - int(proposal.candidate_id.split("-")[-1])
        return CandidateAssessment(valid, None, value, None)

    def generate(current, iteration):
        proposal = _proposal("candidate-{}".format(iteration))
        return (
            replace(
                proposal,
                parent_candidate_id=current.candidate_id,
                overlay=replace(
                    proposal.overlay,
                    parent_candidate_id=current.candidate_id,
                ),
            ),
        )

    bounded = TuningContext(
        inputs=inputs(),
        topology=topology(),
        initial_schedule=two_rank_allreduce_schedule(),
        assess=assess,
        generate=generate,
        max_iterations=25,
        timeout_s=100.0,
        clock=lambda: 0.0,
    )
    result = tune(_candidate(), bounded)

    assert result.iterations == 20
    assert len(result.history) == 21
    assert result.stop_reason == "max_tuning_iterations"

    ticks = iter((0.0, 101.0))
    timed_out = replace(
        bounded,
        max_iterations=1,
        clock=lambda: next(ticks),
    )
    timeout_result = tune(_candidate(), timed_out)

    assert timeout_result.iterations == 0
    assert timeout_result.stop_reason == "verification_timeout"
    assert timeout_result.selected_candidate_id is None


def test_engine_models_reject_invalid_boundaries():
    with pytest.raises(SemanticError):
        OnlinePerformance("invalid", 0.0)
    with pytest.raises(SemanticError):
        OnlinePerformance(0.0, 0.0)

    proposal = _proposal("candidate-boundary")
    for changes in (
        {"candidate_id": ""},
        {"schedule": object()},
        {"overlay": object()},
        {"parent_candidate_id": ""},
        {"tuning_strategy": ()},
        {"tuning_strategy": {1: "invalid"}},
    ):
        with pytest.raises(SemanticError):
            replace(proposal, **changes)

    assessment = CandidateAssessment(_report(), None, 1.0, None)
    for changes in (
        {"report": object()},
        {"artifact": object()},
        {"simulation_time_us": -1.0},
        {"online_performance": object()},
        {"outcome": object()},
    ):
        with pytest.raises(SemanticError):
            replace(assessment, **changes)

    context = _context(
        {"initial": CandidateAssessment(_report(), None, 10.0, None)},
        (),
    )
    for changes in (
        {"inputs": object()},
        {"topology": object()},
        {"initial_schedule": object()},
        {"plan": object()},
        {"initial_schedule": None, "plan": None},
        {"assess": object()},
        {"online_validation": "yes"},
        {"online_performance": {"invalid": object()}},
        {"max_iterations": 0},
        {"timeout_s": 0.0},
        {"clock": object()},
    ):
        with pytest.raises(SemanticError):
            replace(context, **changes)

    result = tune(_candidate(), context)
    entry = result.history[0]
    with pytest.raises(SemanticError):
        replace(entry, candidate_id="")
    with pytest.raises(SemanticError):
        replace(entry, tuning_strategy=())
    with pytest.raises(SemanticError):
        TuningResult(None, None, None, (object(),), "done", 0)
    with pytest.raises(SemanticError):
        replace(result, history=(entry, entry))
    with pytest.raises(SemanticError):
        replace(result, selected_candidate_id="missing")
    with pytest.raises(SemanticError):
        tune(object(), context)
    with pytest.raises(SemanticError):
        tune(_candidate(), object())


def test_engine_rejects_invalid_generator_and_incremental_simulation():
    initial_assessment = CandidateAssessment(_report(), None, 10.0, None)

    def assess(proposal):
        return initial_assessment

    base = TuningContext(
        inputs=inputs(),
        topology=topology(),
        initial_schedule=two_rank_allreduce_schedule(),
        assess=assess,
        generate=lambda current, iteration: (),
        max_iterations=1,
        timeout_s=10.0,
        clock=lambda: 0.0,
    )
    for generate in (
        lambda current, iteration: None,
        lambda current, iteration: (object(),),
        lambda current, iteration: (_proposal("initial"),),
        lambda current, iteration: (
            replace(
                _proposal("wrong-parent"),
                parent_candidate_id="other",
            ),
        ),
    ):
        with pytest.raises(SemanticError):
            tune(_candidate(), replace(base, generate=generate))

    def fail_simulation(schedule):
        raise SemanticError("incremental simulation failed")

    failed = tune(
        _candidate(),
        replace(
            base,
            generate=lambda current, iteration: (
                _proposal("simulation-failed"),
            ),
            simulate=fail_simulation,
        ),
    )
    assert failed.history[-1].rejection_reason == (
        "incremental_simulation_failed"
    )

    invalid_initial = tune(
        _candidate(),
        replace(
            base,
            assess=lambda proposal: CandidateAssessment(
                _report(state=ValidationStatus.INVALID),
                None,
                10.0,
                None,
            ),
        ),
    )
    assert invalid_initial.selected_candidate_id is None
    assert invalid_initial.stop_reason == "no_eligible_initial_candidate"


def test_tuning_budget_includes_initial_and_incremental_validation():
    class Clock:
        def __init__(self, values):
            self.values = iter(values)
            self.last = 0.0

        def __call__(self):
            self.last = next(self.values, self.last)
            return self.last

    initial_assessment = CandidateAssessment(_report(), None, 10.0, None)
    initial_timeout = TuningContext(
        inputs=inputs(),
        topology=topology(),
        initial_schedule=two_rank_allreduce_schedule(),
        assess=lambda proposal: initial_assessment,
        generate=lambda current, iteration: (),
        max_iterations=1,
        timeout_s=10.0,
        clock=Clock((0.0, 11.0)),
    )

    result = tune(_candidate(), initial_timeout)

    assert result.selected_candidate_id is None
    assert result.stop_reason == "verification_timeout"
    assert result.history[0].rejection_reason == "verification_timeout"

    assessment_calls = []

    def assess(proposal):
        assessment_calls.append(proposal.candidate_id)
        return initial_assessment

    incremental_timeout = replace(
        initial_timeout,
        assess=assess,
        generate=lambda current, iteration: (
            _proposal("candidate-timeout"),
        ),
        simulate=lambda schedule: 9.0,
        clock=Clock((0.0, 0.0, 0.0, 0.0, 11.0)),
    )

    result = tune(_candidate(), incremental_timeout)

    assert assessment_calls == ["initial"]
    assert result.stop_reason == "verification_timeout"
    assert result.history[-1].rejection_reason == "verification_timeout"

    generation_timeout = replace(
        incremental_timeout,
        generate=lambda current, iteration: (
            _proposal("candidate-timeout-a"),
            _proposal("candidate-timeout-b"),
        ),
        clock=Clock((0.0, 0.0, 0.0, 11.0)),
    )

    result = tune(_candidate(), generation_timeout)
    timed_out = result.history[1:]

    assert tuple(entry.candidate_id for entry in timed_out) == (
        "candidate-timeout-a",
        "candidate-timeout-b",
    )
    assert all(
        entry.rejection_reason == "verification_timeout"
        for entry in timed_out
    )


def test_invalid_overlay_is_retained_without_stopping_other_candidates():
    invalid_overlay = TuningOverlay(
        overlay_id="invalid-overlay",
        parent_candidate_id="initial",
        channel_count=inputs().solver.max_channels + 1,
    )
    invalid = replace(
        _proposal("candidate-invalid-overlay"),
        overlay=invalid_overlay,
    )
    valid = _proposal("candidate-valid-overlay")
    assessments = {
        "initial": CandidateAssessment(_report(), None, 10.0, None),
        "candidate-valid-overlay": CandidateAssessment(
            _report(),
            None,
            8.0,
            None,
        ),
        "candidate-invalid-overlay": CandidateAssessment(
            _report(),
            None,
            7.0,
            None,
        ),
    }

    result = tune(
        _candidate(),
        _context(assessments, (invalid, valid)),
    )
    by_id = {entry.candidate_id: entry for entry in result.history}

    assert by_id["candidate-invalid-overlay"].rejection_reason == (
        "invalid_tuning_overlay"
    )
    assert result.selected_candidate_id == "candidate-valid-overlay"


def test_tb_order_repair_rejects_inconsistent_hints_and_metadata():
    program, schedule = tb_order_case()
    hint = analyze_tb_order(program, schedule).hints[0]
    for arguments in (
        (object(), program, hint, "overlay"),
        (schedule, object(), hint, "overlay"),
        (schedule, program, object(), "overlay"),
    ):
        with pytest.raises(SemanticError):
            _repair_tb_order(*arguments)
    with pytest.raises(SemanticError, match="missing step"):
        _repair_tb_order(
            schedule,
            program,
            replace(hint, earlier_step_id="missing"),
            "overlay",
        )
    with pytest.raises(SemanticError, match="one transfer"):
        _repair_tb_order(
            schedule,
            program,
            replace(hint, later_step_id=hint.earlier_step_id),
            "overlay",
        )
    with pytest.raises(SemanticError, match="transfer is missing"):
        _repair_tb_order(
            replace(schedule, transfers=schedule.transfers[:-1]),
            program,
            hint,
            "overlay",
        )
    changed_fast = replace(schedule.transfers[-1], channel=1)
    with pytest.raises(SemanticError, match="share a lane"):
        _repair_tb_order(
            replace(
                schedule,
                transfers=schedule.transfers[:-1] + (changed_fast,),
            ),
            program,
            hint,
            "overlay",
        )
    with pytest.raises(SemanticError, match="semantic_predecessors"):
        _repair_tb_order(
            replace(schedule, metadata={"semantic_predecessors": ()}),
            program,
            hint,
            "overlay",
        )
    metadata = dict(schedule.metadata)
    semantic = dict(metadata["semantic_predecessors"])
    semantic["fast"] = ("slow",)
    metadata["semantic_predecessors"] = semantic
    with pytest.raises(SemanticError, match="semantic precedence"):
        _repair_tb_order(
            replace(schedule, metadata=metadata),
            program,
            hint,
            "overlay",
        )
    metadata = dict(schedule.metadata)
    metadata["resource_slots"] = ()
    with pytest.raises(SemanticError, match="resource_slots"):
        _repair_tb_order(
            replace(schedule, metadata=metadata),
            program,
            hint,
            "overlay",
        )
