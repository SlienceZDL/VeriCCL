from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import vericcl.cli.online as online_module
from vericcl.cli.online import build_online_context_factory
from vericcl.errors import InputValidationError
from vericcl.verification.online.pipeline import (
    OnlineContext,
    OnlineStageStatus,
)
from vericcl.verification.online.clock_sync import AlignedTimestamp
from vericcl.verification.online.trace_analysis import (
    PhysicalTransferInterval,
    TraceAnalysis,
)
from vericcl.verification.online.runner import (
    SubprocessCommandExecutor,
    collect_trace_files,
)
from vericcl.verification.pipeline import validate_and_lower_candidate

from tests.unit.verification.helpers import inputs, topology
from vericcl.semantics.collective import CollectiveKind
from tests.unit.xml.helpers import (
    two_rank_allgather_schedule,
    two_rank_allreduce_schedule,
)


pytestmark = pytest.mark.phase07


class FakeExecutor:
    def run(self, request):
        raise AssertionError("injected executor should not run during setup")


def _environment(**changes):
    values = {
        "VERICCL_MSCCL_BUILD_DIR": "/tmp/msccl",
        "VERICCL_NCCL_TESTS_BUILD_DIR": "/tmp/nccl-tests",
        "VERICCL_CLOCK_SYNC_BINARY": "/tmp/vericcl-clock-sync",
        "VERICCL_MPI_LAUNCHER": "/tmp/mpirun",
        "VERICCL_ONLINE_INTER_NODE": "0",
        "VERICCL_MAX_CLOCK_UNCERTAINTY_US": "5.0",
        "VERICCL_CALIBRATION_LINK_CLASS": "intra_node",
        "VERICCL_GPU_MODEL": "V100",
        "VERICCL_NIC_MODEL": "none",
        "VERICCL_CUDA_VERSION": "11.8",
        "VERICCL_NCCL_VERSION": "2.18.5",
        "VERICCL_MSCCL_VERSION": "0.7.4",
        "VERICCL_CALIBRATION_CACHE_PATH": (
            "/tmp/vericcl-unit-calibration-cache.json"
        ),
    }
    values.update(changes)
    return values


def _factory_arguments(tmp_path):
    schedule = two_rank_allreduce_schedule()
    input_value = inputs()
    outcome = validate_and_lower_candidate(
        schedule,
        input_value,
        topology(),
    )
    return (
        outcome.artifact,
        schedule,
        input_value,
        tmp_path / "schedule.xml",
        tmp_path / "traces",
        False,
        30.0,
        False,
    )


def test_online_factory_uses_injected_runtime_dependencies(tmp_path):
    executor = FakeExecutor()
    collector = lambda request: object()

    context = build_online_context_factory(
        _environment(),
        executor=executor,
        trace_collector=collector,
    )(*_factory_arguments(tmp_path))

    assert context.executor is executor
    assert context.trace_collector is collector


def test_online_factory_defaults_to_local_dependencies(tmp_path):
    context = build_online_context_factory(_environment())(
        *_factory_arguments(tmp_path)
    )

    assert isinstance(context.executor, SubprocessCommandExecutor)
    assert context.trace_collector is collect_trace_files


def test_representative_calibration_topology_is_public():
    source = replace(
        topology(),
        node_membership={0: 0, 1: 1},
        gateways=frozenset({0, 1}),
    )

    selected = online_module.representative_calibration_topology(
        source,
        "inter_node",
    )

    assert selected.rank_count == 2
    assert selected.node_membership == {0: 0, 1: 1}


def test_online_factory_requires_explicit_runtime_paths():
    with pytest.raises(InputValidationError, match="MSCCL"):
        build_online_context_factory({})


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"VERICCL_ONLINE_INTER_NODE": "yes"}, "zero or one"),
        ({"VERICCL_MAX_CLOCK_UNCERTAINTY_US": "invalid"}, "numeric"),
        ({"VERICCL_MAX_CLOCK_UNCERTAINTY_US": "-1"}, "non-negative"),
        ({"VERICCL_MPI_LAUNCHER": ""}, "MPI_LAUNCHER"),
        ({"VERICCL_ONLINE_INTER_NODE": "1"}, "MPI_HOSTFILE"),
        ({"VERICCL_CALIBRATION_LINK_CLASS": "invalid"}, "link class"),
        ({"VERICCL_FORCE_RECALIBRATE": "yes"}, "zero or one"),
        ({"VERICCL_TRACE_RECORDS": "0"}, "positive integer"),
    ),
)
def test_online_factory_rejects_invalid_environment(changes, message):
    with pytest.raises(InputValidationError, match=message):
        build_online_context_factory(_environment(**changes))


def test_online_factory_builds_exact_collective_request(tmp_path):
    schedule = two_rank_allreduce_schedule()
    input_value = inputs()
    outcome = validate_and_lower_candidate(
        schedule,
        input_value,
        topology(),
    )
    factory = build_online_context_factory(_environment())

    context = factory(
        outcome.artifact,
        schedule,
        input_value,
        tmp_path / "schedule.xml",
        tmp_path / "traces",
        True,
        30.0,
    )

    assert isinstance(context, OnlineContext)
    assert context.request.kind == input_value.collective.kind
    assert context.request.message_size_bytes == (
        input_value.hyperparameters.total_size_bytes
    )
    assert context.request.datatype == "float"
    assert context.request.reduction_op == "sum"
    assert context.request.inplace is False
    assert context.xml_paths == (tmp_path / "schedule.xml",)
    assert context.trace_file_prefix == tmp_path / "traces" / "msccl-step"
    assert context.online_tuning_requested is True
    assert context.max_clock_uncertainty_us == 5.0
    assert context.request.gpus_per_process == 1
    assert context.mpi_launcher == Path("/tmp/mpirun")
    assert context.trace_record_capacity == 1048576
    assert context.calibration_plan is not None
    assert context.calibration_plan.request.link_class == "intra_node"
    assert context.calibration_plan.request.benchmark_size_bytes == (
        128 * 1024 * 1024
    )

    without_calibration = factory(
        outcome.artifact,
        schedule,
        input_value,
        tmp_path / "schedule.xml",
        tmp_path / "traces-second",
        False,
        30.0,
        False,
    )
    assert without_calibration.calibration_plan is None


def test_operator_launcher_is_independent_from_inter_node_calibration(tmp_path):
    schedule = two_rank_allreduce_schedule()
    input_value = inputs()
    outcome = validate_and_lower_candidate(schedule, input_value, topology())
    hostfile = tmp_path / "hosts"
    hostfile.write_text("node-a\nnode-b\n", encoding="utf-8")
    factory = build_online_context_factory(
        _environment(
            VERICCL_CALIBRATION_LINK_CLASS="inter_node",
            VERICCL_MPI_LAUNCHER="/tmp/mpirun",
            VERICCL_MPI_HOSTFILE=str(hostfile),
            VERICCL_TRACE_RECORDS="2048",
        )
    )

    context = factory(
        outcome.artifact,
        schedule,
        input_value,
        tmp_path / "schedule.xml",
        tmp_path / "traces",
        False,
        30.0,
        False,
    )

    assert context.inter_node is False
    assert context.mpi_launcher == Path("/tmp/mpirun")
    assert context.mpi_hostfile is None
    assert context.request.gpus_per_process == 1
    assert context.trace_record_capacity == 2048


def test_allgather_online_request_matches_expanded_xml_message_size(tmp_path):
    schedule = two_rank_allgather_schedule()
    input_value = inputs(CollectiveKind.ALL_GATHER)
    outcome = validate_and_lower_candidate(
        schedule,
        input_value,
        topology(),
    )
    factory = build_online_context_factory(_environment())

    context = factory(
        outcome.artifact,
        schedule,
        input_value,
        tmp_path / "allgather.xml",
        tmp_path / "traces",
        False,
        1.0,
        False,
    )

    assert context.request.message_size_bytes == (
        input_value.rank_count
        * input_value.hyperparameters.total_size_bytes
    )
    assert 'minBytes="{}"'.format(
        context.request.message_size_bytes
    ) in outcome.artifact.xml_text


def test_calibration_path_signature_hashes_device_and_file_identity(tmp_path):
    hostfile = tmp_path / "hosts"
    hostfile.write_text("node-a\n", encoding="utf-8")
    environment = {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "VERICCL_MPI_HOSTFILE": str(hostfile),
    }

    first = dict(online_module._calibration_path_variables(environment))
    hostfile.write_text("node-b\n", encoding="utf-8")
    second = dict(online_module._calibration_path_variables(environment))

    assert first["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert first["VERICCL_MPI_HOSTFILE"] != second["VERICCL_MPI_HOSTFILE"]


def test_calibration_plan_measures_full_waves_with_generated_xml(
    tmp_path,
    monkeypatch,
):
    schedule = two_rank_allreduce_schedule()
    input_value = inputs()
    input_value = replace(
        input_value,
        hyperparameters=replace(
            input_value.hyperparameters,
            total_size_bytes=128 * 1024 * 1024,
            slice_size_bytes=64 * 1024 * 1024,
            max_calibration_channels=2,
        ),
    )
    outcome = validate_and_lower_candidate(
        schedule,
        inputs(),
        topology(),
    )
    captured = []

    def measured(context):
        captured.append(context)
        intervals = tuple(
            PhysicalTransferInterval(
                transfer_id="calibration-send-{:08d}".format(logical),
                iteration=iteration,
                send=None,
                receive=None,
                local=None,
                physical_start=AlignedTimestamp(0.0, 0.0),
                physical_end=AlignedTimestamp(5.0, 0.0),
                endpoint_order_uncertain=False,
                sender_start=AlignedTimestamp(0.0, 0.0),
                sender_end=AlignedTimestamp(5.0, 0.0),
            )
            for iteration in range(20)
            for logical in range(2)
        )
        return SimpleNamespace(
            release_status=OnlineStageStatus.PASSED,
            online_operator_validation=OnlineStageStatus.PASSED,
            trace_analysis=TraceAnalysis(intervals, (), (), (), True),
            failure_message=None,
        )

    monkeypatch.setattr(online_module, "run_online_validation", measured)
    ticks = iter((100.0, 100.2, 100.4))
    monkeypatch.setattr(
        online_module,
        "_monotonic",
        lambda: next(ticks),
        raising=False,
    )
    factory = build_online_context_factory(
        _environment(
            VERICCL_CALIBRATION_CACHE_PATH=str(tmp_path / "cache.json")
        )
    )
    context = factory(
        outcome.artifact,
        schedule,
        input_value,
        tmp_path / "schedule.xml",
        tmp_path / "traces",
        False,
        1.0,
    )

    first = context.calibration_plan.measure_point(
        context.calibration_plan.signatures[0]
    )
    second = context.calibration_plan.measure_point(
        context.calibration_plan.signatures[1]
    )
    repeated = context.calibration_plan.measure_point(
        context.calibration_plan.signatures[0]
    )

    assert first.concurrency == 1
    assert second.concurrency == 2
    assert repeated.concurrency == 1
    assert second.duration_statistics.p95_us == 5.0
    assert captured[0].request.kind.value == "broadcast"
    assert captured[0].request.gpus_per_process == 1
    assert captured[0].single_process_release_validation is True
    assert captured[0].timeout_s == pytest.approx(0.5)
    assert captured[0].xml_paths[0].is_file()
    assert captured[1].timeout_s == pytest.approx(0.8)
