from dataclasses import replace
import hashlib
from pathlib import Path

import pytest
import vericcl.verification.online.pipeline as pipeline_module

from vericcl.errors import SemanticError
from vericcl.semantics.collective import CollectiveKind
from vericcl.topology import load_topology
from vericcl.verification.online.cache import (
    CalibrationCache,
    EnvironmentSignature,
)
from vericcl.verification.online.calibration import (
    CalibrationPoint,
    CalibrationRequest,
)
from vericcl.verification.online.model import NcclTestRequest
from vericcl.verification.online.pipeline import (
    CalibrationPlan,
    OnlineContext,
    OnlineStageStatus,
    TraceCollectionResult,
    attach_online_result_to_tuning_context,
    run_online_validation,
)
from vericcl.verification.online.runner import (
    ProcessRequest,
    ProcessResult,
    collect_trace_files,
)
from vericcl.verification.online.statistics import summarize_runs
from vericcl.verification.online.trace_analysis import TraceAnalysis
from vericcl.verification.online.trace_format import (
    RawStepTraceRecord,
    encode_raw_trace,
)
from vericcl.xml.lower import lower_to_xml
from vericcl.tuning.engine import TuningContext

from tests.unit.xml.helpers import resolved, two_rank_allreduce_schedule


pytestmark = pytest.mark.phase06


HELP_TEXT = "usage: perf -b -e -w -n -c -g -d -o -r"


class FakeExecutor:
    def __init__(self, *, fail_release=False, trace_time_us=999.0):
        self.calls = []
        self.fail_release = fail_release
        self.trace_time_us = trace_time_us

    def run(self, request: ProcessRequest) -> ProcessResult:
        self.calls.append(request)
        if request.command[-1:] == ("--help",):
            return ProcessResult(0, HELP_TEXT, "")
        if Path(request.command[-2]).name == "vericcl_clock_sync":
            return ProcessResult(0, _clock_output(), "")
        trace_enabled = request.environment["VERICCL_TRACE_ENABLE"] == "1"
        if self.fail_release and not trace_enabled:
            return ProcessResult(2, "", "release failed")
        time_us = self.trace_time_us if trace_enabled else 10.0
        return ProcessResult(0, _performance_output(time_us), "")


def _performance_output(time_us):
    return (
        "1024 256 float sum -1 "
        "{} 100.0 150.0 0 {} 90.0 140.0 0"
    ).format(time_us, time_us + 1.0)


def _clock_output():
    rows = []
    for rank in (0, 1):
        for index in range(3):
            ticks = 1000 + index * 100
            host = 1000000 + index * 100
            rows.append(
                "VERICCL_CLOCK_SYNC {} {} {} {} 0 0".format(
                    rank,
                    ticks,
                    host * 1000,
                    host * 1000,
                )
            )
    return "\n".join(rows)


def _analysis(*, tuning_eligible=True):
    return TraceAnalysis(
        intervals=(),
        step_waits=(),
        bottlenecks=(),
        uncertain_comparisons=(
            () if tuning_eligible else ("uncertain",)
        ),
        tuning_eligible=tuning_eligible,
    )


def _collector(
    *,
    complete=True,
    tuning_eligible=True,
    error=None,
    clock_uncertainty_us=0.0,
):
    def collect(request):
        if error is not None:
            raise error
        assert request.rank_count == 2
        assert request.clock_sync_output
        return TraceCollectionResult(
            analysis=_analysis(tuning_eligible=tuning_eligible),
            rank_files=(
                Path("trace.rank-0.bin"),
                Path("trace.rank-1.bin"),
            ),
            complete=complete,
            clock_uncertainty_us=clock_uncertainty_us,
        )

    return collect


def _context(tmp_path, **changes):
    tmp_path.mkdir(parents=True, exist_ok=True)
    schedule = two_rank_allreduce_schedule()
    inputs = resolved(CollectiveKind.ALL_REDUCE, ranks=2, slices=1)
    artifact = lower_to_xml(schedule, inputs, load_topology(inputs))
    xml_path = tmp_path / "allreduce.xml"
    xml_path.write_text(artifact.xml_text, encoding="utf-8")
    binary_directory = tmp_path / "nccl-tests"
    binary_directory.mkdir()
    binary = binary_directory / "all_reduce_perf"
    binary.write_text("binary", encoding="utf-8")
    binary.chmod(0o755)
    library_directory = tmp_path / "msccl-lib"
    library_directory.mkdir()
    mpi = tmp_path / "mpirun"
    mpi.write_text("binary", encoding="utf-8")
    mpi.chmod(0o755)
    clock = tmp_path / "vericcl_clock_sync"
    clock.write_text("binary", encoding="utf-8")
    clock.chmod(0o755)
    values = {
        "artifact": artifact,
        "schedule": schedule,
        "inputs": inputs,
        "request": NcclTestRequest(
            kind=CollectiveKind.ALL_REDUCE,
            message_size_bytes=1024,
            datatype="float",
            reduction_op="sum",
            root=None,
            inplace=False,
            binary_directory=str(binary_directory),
        ),
        "xml_paths": (xml_path,),
        "msccl_library_path": library_directory,
        "executor": FakeExecutor(),
        "environment": {"VERICCL_TEST_VALUE": "preserved"},
        "inter_node": False,
        "mpi_launcher": mpi,
        "mpi_hostfile": None,
        "trace_file_prefix": tmp_path / "traces" / "step",
        "clock_sync_binary": clock,
        "max_clock_uncertainty_us": 10.0,
        "trace_collector": _collector(),
        "online_tuning_requested": True,
    }
    values.update(changes)
    return OnlineContext(**values)


def _signature(concurrency=1):
    return EnvironmentSignature(
        link_class="intra_node",
        topology_signature="a" * 64,
        gpu_model="V100",
        nic_model="none",
        cuda_version="11.8",
        nccl_version="2.18.5",
        msccl_version="0.7.4",
        protocol="Simple",
        slice_size_bytes=1024,
        benchmark_size_bytes=128 * 1024 * 1024,
        concurrency=concurrency,
        nccl_buffsize_bytes=2048,
        chunk_steps=4,
        slice_steps=4,
        path_variables=(("LD_LIBRARY_PATH", "/opt/msccl/lib"),),
    )


def _point(*, concurrency=1, stable=True, duration=10.0):
    samples = (
        (duration,) * 20
        if stable
        else tuple(1.0 if index % 2 else 20.0 for index in range(20))
    )
    return CalibrationPoint(
        concurrency=concurrency,
        duration_statistics=summarize_runs(samples),
        full_wave_count=(128 * 1024 * 1024 // 1024) // concurrency,
        tail_transfer_count=(128 * 1024 * 1024 // 1024) % concurrency,
    )


def _calibration_plan(cache, measure, *, force=False):
    return CalibrationPlan(
        request=CalibrationRequest(
            link_class="intra_node",
            slice_size_bytes=1024,
            max_calibration_channels=1,
            datatype="float",
        ),
        alpha_us=1.0,
        signatures=(_signature(),),
        cache=cache,
        measure_point=measure,
        force_recalibrate=force,
    )


def test_calibration_plan_accepts_complete_global_software_range():
    CalibrationPlan(
        request=CalibrationRequest(
            link_class="intra_node",
            slice_size_bytes=1024,
            max_calibration_channels=32,
            datatype="float",
        ),
        alpha_us=1.0,
        signatures=tuple(_signature(value) for value in range(1, 17)),
        cache=CalibrationCache(),
        measure_point=lambda signature: _point(
            concurrency=signature.concurrency,
        ),
    )


def _performance_calls(executor):
    return [
        call
        for call in executor.calls
        if call.command[-1:] != ("--help",)
        and not any(
            Path(argument).name == "vericcl_clock_sync"
            for argument in call.command
        )
    ]


def test_runtime_incompatibility_blocks_every_process_launch(tmp_path):
    executor = FakeExecutor()
    base = _context(tmp_path, executor=executor)
    context = replace(
        base,
        artifact=replace(base.artifact, runtime_compatible=False),
    )

    result = run_online_validation(context)

    assert result.preflight_status is OnlineStageStatus.FAILED
    assert result.failure_code == "runtime_incompatible"
    assert executor.calls == []


def test_missing_nccl_tests_binary_blocks_launch(tmp_path):
    executor = FakeExecutor()
    context = _context(tmp_path, executor=executor)
    Path(context.request.binary_directory, "all_reduce_perf").unlink()

    result = run_online_validation(context)

    assert result.preflight_status is OnlineStageStatus.FAILED
    assert result.failure_code == "nccl_tests_binary_missing"
    assert executor.calls == []


@pytest.mark.parametrize(
    ("field", "value", "code"),
    (
        ("message_size_bytes", 2048, "message_size_mismatch"),
        ("datatype", "half", "datatype_mismatch"),
        ("reduction_op", "prod", "reduction_op_mismatch"),
        ("inplace", True, "inplace_mismatch"),
    ),
)
def test_preflight_rejects_request_semantic_mismatch(
    tmp_path,
    field,
    value,
    code,
):
    executor = FakeExecutor()
    base = _context(tmp_path, executor=executor)
    context = replace(base, request=replace(base.request, **{field: value}))

    result = run_online_validation(context)

    assert result.failure_code == code
    assert executor.calls == []


def test_preflight_requires_one_matching_xml_and_inter_node_mpi(tmp_path):
    base = _context(tmp_path)

    multiple = run_online_validation(
        replace(base, xml_paths=base.xml_paths + base.xml_paths)
    )
    missing_mpi = run_online_validation(
        replace(base, inter_node=True, mpi_launcher=None)
    )

    assert multiple.failure_code == "xml_path_count_invalid"
    assert missing_mpi.failure_code == "mpi_launcher_missing"


def test_preflight_requires_mpi_for_intra_node_one_process_per_gpu(tmp_path):
    context = _context(tmp_path, mpi_launcher=None)

    result = run_online_validation(context)

    assert result.failure_code == "mpi_launcher_missing"


def test_two_rank_inter_node_launcher_maps_one_process_per_node(tmp_path):
    executor = FakeExecutor()
    base = _context(tmp_path, executor=executor)
    hostfile = tmp_path / "hosts"
    hostfile.write_text("node-a slots=1\nnode-b slots=1\n", encoding="utf-8")
    context = replace(
        base,
        inter_node=True,
        mpi_hostfile=hostfile,
        environment={
            **base.environment,
            "VERICCL_MPI_TCP_IF_INCLUDE": "10.0.0.0/24",
        },
    )

    result = run_online_validation(context)

    assert result.online_operator_validation is OnlineStageStatus.PASSED
    command = _performance_calls(executor)[0].command
    assert command[:19] == (
        str(context.mpi_launcher),
        "-np",
        "2",
        "-N",
        "1",
        "--hostfile",
        str(hostfile),
        "-mca",
        "pml",
        "ob1",
        "-mca",
        "btl",
        "tcp,self,vader",
        "-mca",
        "btl_vader_single_copy_mechanism",
        "none",
        "-mca",
        "btl_tcp_if_include",
        "10.0.0.0/24",
    )
    assert "OLDPWD" not in command


def test_preflight_rejects_conflicting_runtime_environment(tmp_path):
    context = _context(
        tmp_path,
        environment={"NCCL_ALGO": "Ring"},
    )

    result = run_online_validation(context)

    assert result.failure_code == "runtime_environment_conflict"


def test_release_failure_prevents_trace(tmp_path):
    executor = FakeExecutor(fail_release=True)
    context = _context(tmp_path, executor=executor)

    result = run_online_validation(context)

    assert result.release_status is OnlineStageStatus.FAILED
    assert result.online_operator_validation is OnlineStageStatus.NOT_RUN
    assert result.failure_code == "release_measurement_failed"
    assert len(_performance_calls(executor)) == 1


def test_trace_failure_preserves_successful_release_measurement(tmp_path):
    executor = FakeExecutor()
    context = _context(
        tmp_path,
        executor=executor,
        trace_collector=_collector(error=SemanticError("trace incomplete")),
    )

    result = run_online_validation(context)

    assert result.release_status is OnlineStageStatus.PASSED
    assert result.release_history.stable
    assert result.online_operator_validation is OnlineStageStatus.FAILED
    assert result.failure_code == "trace_collection_failed"
    assert result.online_tuning_allowed is False
    assert len(_performance_calls(executor)) == 21


def test_release_and_trace_runs_are_separate_and_share_exact_parameters(tmp_path):
    executor = FakeExecutor(trace_time_us=999.0)
    context = _context(tmp_path, executor=executor)

    result = run_online_validation(context)

    calls = _performance_calls(executor)
    release_calls = [
        call
        for call in calls
        if call.environment["VERICCL_TRACE_ENABLE"] == "0"
    ]
    trace_calls = [
        call
        for call in calls
        if call.environment["VERICCL_TRACE_ENABLE"] == "1"
    ]
    assert result.release_status is OnlineStageStatus.PASSED
    assert result.online_operator_validation is OnlineStageStatus.PASSED
    assert result.release_history.all_samples_us == (10.0,) * 20
    assert len(release_calls) == 20
    assert len(trace_calls) == 1
    def timing_neutral(command):
        normalized = list(command)
        normalized[normalized.index("-w") + 1] = "<warmup>"
        normalized[normalized.index("-c") + 1] = "<checks>"
        return tuple(normalized)

    assert {
        timing_neutral(call.command) for call in calls
    } == {timing_neutral(calls[0].command)}
    assert release_calls[0].command[
        release_calls[0].command.index("-w") + 1
    ] == "5"
    assert release_calls[0].command[
        release_calls[0].command.index("-c") + 1
    ] == "1"
    assert trace_calls[0].command[
        trace_calls[0].command.index("-w") + 1
    ] == "0"
    assert trace_calls[0].command[
        trace_calls[0].command.index("-c") + 1
    ] == "0"
    assert all(
        call.environment["NCCL_ALGO"] == "MSCCL,RING"
        for call in calls
    )
    assert all(call.environment["NCCL_BUFFSIZE"] == "2048" for call in calls)
    assert all(
        call.environment["VERICCL_EXPECTED_MSCCL_CHUNKSTEPS"] == "4"
        and call.environment["VERICCL_EXPECTED_MSCCL_SLICESTEPS"] == "4"
        and call.environment["VERICCL_TEST_VALUE"] == "preserved"
        for call in calls
    )
    assert result.online_tuning_allowed is True
    assert result.tuning_evidence is not None


def test_trace_capacity_is_raised_to_cover_both_timing_blocks(tmp_path):
    executor = FakeExecutor()
    context = _context(
        tmp_path,
        executor=executor,
        trace_record_capacity=1,
    )

    result = run_online_validation(context)

    entries_per_rank = max(
        sum(
            1
            for step in context.artifact.tb_program.steps_by_id.values()
            if step.rank == rank
        )
        for rank in range(context.inputs.rank_count)
    )
    assert int(result.runtime_environment["VERICCL_TRACE_RECORDS"]) >= (
        42 * entries_per_rank
    )


def test_incomplete_or_uncertain_trace_cannot_drive_online_tuning(tmp_path):
    incomplete = run_online_validation(
        _context(tmp_path / "incomplete", trace_collector=_collector(complete=False))
    )
    uncertain = run_online_validation(
        _context(
            tmp_path / "uncertain",
            trace_collector=_collector(tuning_eligible=False),
        )
    )

    assert incomplete.online_operator_validation is OnlineStageStatus.FAILED
    assert incomplete.online_tuning_allowed is False
    assert uncertain.online_operator_validation is OnlineStageStatus.PASSED
    assert uncertain.online_tuning_allowed is False
    assert uncertain.tuning_evidence is None

    excessive_uncertainty = run_online_validation(
        _context(
            tmp_path / "excessive",
            trace_collector=_collector(clock_uncertainty_us=11.0),
        )
    )
    assert excessive_uncertainty.failure_code == "trace_collection_failed"


def test_stable_calibration_requests_resolve_without_mutating_schedule(tmp_path):
    measured = []
    cache = CalibrationCache()
    context = _context(
        tmp_path,
        calibration_plan=_calibration_plan(
            cache,
            lambda signature: measured.append(signature.concurrency)
            or _point(),
        ),
    )

    result = run_online_validation(context)

    assert measured == [1]
    assert result.calibration_status is OnlineStageStatus.REQUIRES_RESOLVE
    assert result.requires_resolve is True
    assert result.release_status is OnlineStageStatus.NOT_RUN
    assert result.calibration.curve.invbw_us == pytest.approx(10.0)
    assert context.schedule is result.context_schedule


def test_calibration_cache_reuse_and_force_recalibration(tmp_path):
    cache = CalibrationCache()
    cache.put(_signature(), _point(duration=10.0))
    reused = run_online_validation(
        _context(
            tmp_path / "reuse",
            calibration_plan=_calibration_plan(
                cache,
                lambda signature: pytest.fail("cache was not reused"),
            ),
        )
    )
    measured = []
    forced = run_online_validation(
        _context(
            tmp_path / "force",
            calibration_plan=_calibration_plan(
                cache,
                lambda signature: measured.append(signature.concurrency)
                or _point(duration=12.0),
                force=True,
            ),
        )
    )

    assert reused.calibration.cache_hit_concurrencies == (1,)
    assert reused.requires_resolve is True
    assert measured == [1]
    assert forced.calibration.cache_hit_concurrencies == ()
    assert forced.calibration.curve.invbw_us == pytest.approx(12.0)


def test_unstable_calibration_blocks_tuning_but_not_operator_validation(tmp_path):
    executor = FakeExecutor()
    context = _context(
        tmp_path,
        executor=executor,
        calibration_plan=_calibration_plan(
            CalibrationCache(),
            lambda signature: _point(stable=False),
        ),
    )

    result = run_online_validation(context)

    assert result.calibration_status is OnlineStageStatus.UNSTABLE
    assert result.release_status is OnlineStageStatus.PASSED
    assert result.online_operator_validation is OnlineStageStatus.PASSED
    assert result.online_tuning_allowed is False


def test_calibration_elapsed_time_reduces_shared_operator_budget(
    tmp_path,
    monkeypatch,
):
    context = _context(
        tmp_path,
        timeout_s=5.0,
        calibration_plan=_calibration_plan(
            CalibrationCache(),
            lambda signature: _point(stable=False),
        ),
    )
    ticks = iter((0.0, 6.0))
    monkeypatch.setattr(
        pipeline_module,
        "_monotonic",
        lambda: next(ticks),
    )

    result = run_online_validation(context)

    assert result.release_status is OnlineStageStatus.FAILED
    assert result.failure_code == "online_timeout"
    assert result.requires_resolve is False


def test_online_context_rejects_invalid_process_dependencies(tmp_path):
    context = _context(tmp_path)
    with pytest.raises(SemanticError, match="executor"):
        replace(context, executor=object())
    with pytest.raises(SemanticError, match="chunk_steps"):
        replace(context, chunk_steps=2)
    with pytest.raises(SemanticError, match="max_clock_uncertainty_us"):
        replace(context, max_clock_uncertainty_us=float("inf"))


@pytest.mark.parametrize(
    ("case", "code"),
    (
        ("missing_xml", "xml_path_missing"),
        ("changed_xml", "xml_artifact_mismatch"),
        ("missing_library", "msccl_library_missing"),
        ("missing_hostfile", "mpi_hostfile_missing"),
        ("missing_trace_prefix", "trace_prefix_missing"),
        ("missing_clock_binary", "clock_sync_binary_missing"),
        ("geometry", "schedule_input_mismatch"),
    ),
)
def test_preflight_reports_precise_missing_prerequisite(tmp_path, case, code):
    context = _context(tmp_path)
    if case == "missing_xml":
        context.xml_paths[0].unlink()
    elif case == "changed_xml":
        context.xml_paths[0].write_text("<different/>\n", encoding="utf-8")
    elif case == "missing_library":
        context.msccl_library_path.rmdir()
    elif case == "missing_hostfile":
        context = replace(context, mpi_hostfile=tmp_path / "missing-hostfile")
    elif case == "missing_trace_prefix":
        context = replace(context, trace_file_prefix=None)
    elif case == "missing_clock_binary":
        context = replace(context, clock_sync_binary=None)
    elif case == "geometry":
        context = replace(
            context,
            inputs=replace(context.inputs, rank_count=3),
        )

    result = run_online_validation(context)

    assert result.failure_code == code


def test_pipeline_rejects_direct_intra_node_launch_without_mpi(tmp_path):
    executor = FakeExecutor()
    context = _context(tmp_path, executor=executor, mpi_launcher=None)

    result = run_online_validation(context)

    assert result.failure_code == "mpi_launcher_missing"
    assert executor.calls == []


def test_clock_and_trace_process_failures_preserve_release_result(tmp_path):
    class StageFailureExecutor(FakeExecutor):
        def __init__(self, failed_stage):
            super().__init__()
            self.failed_stage = failed_stage

        def run(self, request):
            is_clock = any(
                Path(argument).name == "vericcl_clock_sync"
                for argument in request.command
            )
            is_trace = request.environment.get("VERICCL_TRACE_ENABLE") == "1"
            if self.failed_stage == "clock" and is_clock:
                self.calls.append(request)
                return ProcessResult(2, "", "clock failed")
            if self.failed_stage == "trace" and is_trace:
                self.calls.append(request)
                return ProcessResult(2, "", "trace failed")
            return super().run(request)

    clock = run_online_validation(
        _context(tmp_path / "clock", executor=StageFailureExecutor("clock"))
    )
    trace = run_online_validation(
        _context(tmp_path / "trace", executor=StageFailureExecutor("trace"))
    )

    assert clock.release_status is OnlineStageStatus.PASSED
    assert clock.failure_code == "clock_sync_failed"
    assert trace.release_status is OnlineStageStatus.PASSED
    assert trace.failure_code == "trace_run_failed"


def test_invalid_trace_collector_result_is_rejected(tmp_path):
    result = run_online_validation(
        _context(tmp_path, trace_collector=lambda request: object())
    )

    assert result.failure_code == "trace_collection_failed"
    assert result.online_operator_validation is OnlineStageStatus.FAILED


def test_invalid_calibration_result_blocks_tuning_but_not_trace(tmp_path):
    context = _context(
        tmp_path,
        calibration_plan=_calibration_plan(
            CalibrationCache(),
            lambda signature: object(),
        ),
    )

    result = run_online_validation(context)

    assert result.calibration_status is OnlineStageStatus.FAILED
    assert result.online_operator_validation is OnlineStageStatus.PASSED
    assert result.online_tuning_allowed is False
    assert result.failure_code == "calibration_failed"


def test_unstable_release_statistics_do_not_drive_online_tuning(tmp_path):
    class UnstableExecutor(FakeExecutor):
        def __init__(self):
            super().__init__()
            self.release_index = 0

        def run(self, request):
            is_release = (
                request.command[-1:] != ("--help",)
                and request.environment.get("VERICCL_TRACE_ENABLE") == "0"
                and not any(
                    Path(argument).name == "vericcl_clock_sync"
                    for argument in request.command
                )
            )
            if is_release:
                self.calls.append(request)
                self.release_index += 1
                value = 1.0 if self.release_index % 2 else 20.0
                return ProcessResult(0, _performance_output(value), "")
            return super().run(request)

    result = run_online_validation(
        _context(tmp_path, executor=UnstableExecutor())
    )

    assert result.release_status is OnlineStageStatus.UNSTABLE
    assert len(result.release_history.rounds) == 3
    assert result.online_operator_validation is OnlineStageStatus.PASSED
    assert result.online_tuning_allowed is False


@pytest.mark.parametrize(
    ("old", "new", "code"),
    (
        ('coll="allreduce"', 'coll="broadcast"', "xml_collective_mismatch"),
        ('inplace="0"', 'inplace="1"', "xml_inplace_mismatch"),
        ('proto="Simple"', 'proto="LL"', "xml_protocol_mismatch"),
        ('ngpus="2"', 'ngpus="3"', "xml_rank_count_mismatch"),
        ('maxBytes="1025"', 'maxBytes="1026"', "xml_size_range_mismatch"),
    ),
)
def test_preflight_validates_exact_xml_runtime_attributes(
    tmp_path,
    old,
    new,
    code,
):
    context = _context(tmp_path)
    xml_text = context.artifact.xml_text.replace(old, new, 1)
    artifact = replace(
        context.artifact,
        xml_text=xml_text,
        sha256=hashlib.sha256(xml_text.encode("utf-8")).hexdigest(),
    )
    context.xml_paths[0].write_text(xml_text, encoding="utf-8")

    result = run_online_validation(replace(context, artifact=artifact))

    assert result.failure_code == code


def _write_trace_and_collect(request):
    for rank in range(request.rank_count):
        records = []
        for iteration in range(21):
            for entry in sorted(
                request.sidecar.entries.values(),
                key=lambda item: item.key,
            ):
                if entry.rank != rank:
                    continue
                records.append(
                    RawStepTraceRecord(
                        rank=rank,
                        tb_id=entry.tb_id,
                        step_index=entry.step_index,
                        endpoint_type=entry.runtime_endpoint_type,
                        peer=entry.peer,
                        channel=entry.runtime_channel,
                        iteration=iteration,
                        tb_reach=10,
                        dependency_done=20,
                        transfer_start=30,
                        transfer_end=40,
                        flags=0,
                        reserved=0,
                    )
                )
        Path("{}.rank-{}.bin".format(request.file_prefix, rank)).write_bytes(
            encode_raw_trace(records, rank=rank)
        )
    return collect_trace_files(request)


def test_complete_trace_waits_are_forwarded_as_tuning_evidence(tmp_path):
    result = run_online_validation(
        _context(tmp_path, trace_collector=_write_trace_and_collect)
    )

    assert result.online_tuning_allowed is True
    assert result.tuning_evidence.wait_us_by_transfer
    assert result.tuning_evidence.bottleneck_priorities


def test_valid_online_result_is_attached_to_candidate_tuning_context(tmp_path):
    online_context = _context(tmp_path)
    result = run_online_validation(online_context)
    tuning_context = TuningContext(
        inputs=online_context.inputs,
        topology=load_topology(online_context.inputs),
        initial_schedule=online_context.schedule,
    )

    updated = attach_online_result_to_tuning_context(
        tuning_context,
        "candidate-initial",
        result,
    )

    assert updated.online_validation is True
    assert updated.online_performance[
        "candidate-initial"
    ].median_time_us == pytest.approx(10.0)
    assert (
        updated.online_trace_evidence["candidate-initial"]
        is result.tuning_evidence
    )

    invalid = run_online_validation(
        _context(
            tmp_path / "invalid",
            trace_collector=_collector(complete=False),
        )
    )
    with pytest.raises(SemanticError, match="not eligible"):
        attach_online_result_to_tuning_context(
            tuning_context,
            "candidate-invalid",
            invalid,
        )
    with pytest.raises(SemanticError, match="TuningContext"):
        attach_online_result_to_tuning_context(
            object(),
            "candidate-invalid",
            result,
        )
    with pytest.raises(SemanticError, match="candidate_id"):
        attach_online_result_to_tuning_context(
            tuning_context,
            "",
            result,
        )
    with pytest.raises(SemanticError, match="OnlineValidationResult"):
        attach_online_result_to_tuning_context(
            tuning_context,
            "candidate-invalid",
            object(),
        )


def test_calibration_plan_rejects_incomplete_or_inconsistent_contract():
    plan = _calibration_plan(
        CalibrationCache(),
        lambda signature: _point(),
    )
    for changes in (
        {"alpha_us": float("inf")},
        {"signatures": ()},
        {
            "signatures": (
                replace(plan.signatures[0], protocol="LL"),
            )
        },
        {"cache": object()},
        {"measure_point": object()},
        {"force_recalibrate": "yes"},
    ):
        with pytest.raises(SemanticError):
            replace(plan, **changes)
