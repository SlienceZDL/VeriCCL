import math
from dataclasses import replace
from pathlib import Path

import pytest

from vericcl.errors import SemanticError
from vericcl.semantics.collective import CollectiveKind
from vericcl.verification.online.model import (
    NcclTestMeasurement,
    NcclTestRequest,
    PerformanceStatistics,
)
from vericcl.verification.online.nccl_tests import (
    NcclTestsHelpValidator,
    build_nccl_tests_command,
    parse_nccl_tests_output,
)


pytestmark = pytest.mark.phase06


def _request(kind, **changes):
    values = {
        "kind": kind,
        "message_size_bytes": 256 * 1024 * 1024,
        "datatype": "float",
        "reduction_op": None,
        "root": None,
        "inplace": False,
        "binary_directory": "/opt/nccl-tests/build",
    }
    values.update(changes)
    return NcclTestRequest(**values)


def test_allreduce_command_uses_exact_size_and_statistics_counts():
    command = build_nccl_tests_command(
        _request(
            CollectiveKind.ALL_REDUCE,
            reduction_op="sum",
        )
    )

    assert command[0] == str(
        Path("/opt/nccl-tests/build/all_reduce_perf")
    )
    assert command[1:3] == ("-g", "1")
    assert command[-14:] == (
        "-b",
        "268435456",
        "-e",
        "268435456",
        "-w",
        "5",
        "-n",
        "20",
        "-c",
        "1",
        "-d",
        "float",
        "-o",
        "sum",
    )


def test_broadcast_command_sets_exact_root():
    command = build_nccl_tests_command(
        _request(
            CollectiveKind.BROADCAST,
            message_size_bytes=128 * 1024 * 1024,
            root=0,
        )
    )

    assert command[0].endswith("/broadcast_perf")
    assert command[-2:] == ("-r", "0")


@pytest.mark.parametrize(
    ("kind", "binary"),
    (
        (CollectiveKind.BROADCAST, "broadcast_perf"),
        (CollectiveKind.REDUCE, "reduce_perf"),
        (CollectiveKind.ALL_GATHER, "all_gather_perf"),
        (CollectiveKind.ALL_REDUCE, "all_reduce_perf"),
        (CollectiveKind.ALL_TO_ALL, "alltoall_perf"),
        (CollectiveKind.REDUCE_SCATTER, "reduce_scatter_perf"),
    ),
)
def test_six_online_collectives_map_to_exact_binary(kind, binary):
    changes = {}
    if kind in {CollectiveKind.REDUCE, CollectiveKind.BROADCAST}:
        changes["root"] = 0
    if kind in {CollectiveKind.REDUCE, CollectiveKind.ALL_REDUCE,
                CollectiveKind.REDUCE_SCATTER}:
        changes["reduction_op"] = "sum"

    command = build_nccl_tests_command(_request(kind, **changes))

    assert Path(command[0]).name == binary


def _output_row(
    out_time,
    in_time,
    *,
    size=268435456,
    out_wrong=0,
    in_wrong=0,
):
    return (
        "{} 67108864 float sum -1 "
        "{} 100.0 150.0 {} {} 90.0 140.0 {}"
    ).format(
        size,
        out_time,
        out_wrong,
        in_time,
        in_wrong,
    )


def test_parser_retains_out_of_place_and_in_place_measurements():
    text = "\n".join(
        (
            "# nccl-tests header",
            "# size count type redop root time algbw busbw #wrong "
            "time algbw busbw #wrong",
            _output_row(10.5, 11.5),
        )
    )

    runs = parse_nccl_tests_output(text, 268435456)

    assert len(runs) == 1
    assert runs[0].message_size_bytes == 268435456
    assert runs[0].datatype == "float"
    assert runs[0].out_of_place.time_us == pytest.approx(10.5)
    assert runs[0].in_place.time_us == pytest.approx(11.5)
    assert runs[0].selected_time_us(inplace=False) == pytest.approx(10.5)
    assert runs[0].selected_time_us(inplace=True) == pytest.approx(11.5)


def test_parser_accepts_twenty_independent_process_rows():
    text = "\n".join(
        _output_row(float(index), float(index) + 0.5)
        for index in range(1, 21)
    )

    runs = parse_nccl_tests_output(text, 268435456)

    assert len(runs) == 20
    assert tuple(run.out_of_place.time_us for run in runs) == tuple(
        float(index) for index in range(1, 21)
    )


def test_parser_rejects_wrong_size_and_correctness_failures():
    with pytest.raises(SemanticError, match="message size"):
        parse_nccl_tests_output(
            _output_row(10.0, 11.0, size=134217728),
            268435456,
        )
    with pytest.raises(SemanticError, match="correctness"):
        parse_nccl_tests_output(
            _output_row(10.0, 11.0, out_wrong=1),
            268435456,
        )
    with pytest.raises(SemanticError, match="performance row"):
        parse_nccl_tests_output("# no measurements", 268435456)


def test_help_validator_caches_binary_help_and_rejects_missing_options():
    request = _request(
        CollectiveKind.BROADCAST,
        root=0,
    )
    calls = []
    all_options = "usage: perf -b -e -w -n -c -g -d -o -r"

    validator = NcclTestsHelpValidator()
    validator.validate(
        request,
        lambda command: calls.append(command) or all_options,
    )
    validator.validate(
        request,
        lambda command: calls.append(command) or all_options,
    )

    assert calls == [
        ("/opt/nccl-tests/build/broadcast_perf", "--help")
    ]

    with pytest.raises(SemanticError, match="-r"):
        NcclTestsHelpValidator().validate(
            request,
            lambda command: "usage: perf -b -e -w -n -c -g -d",
        )


def test_request_rejects_unsupported_or_inconsistent_collectives():
    with pytest.raises(SemanticError, match="online collective"):
        _request(CollectiveKind.SCATTER)
    with pytest.raises(SemanticError, match="reduction_op"):
        _request(CollectiveKind.ALL_REDUCE)
    with pytest.raises(SemanticError, match="root"):
        _request(CollectiveKind.BROADCAST)
    with pytest.raises(SemanticError, match="must not define root"):
        _request(CollectiveKind.ALL_GATHER, root=0)


def test_request_and_command_reject_invalid_boundaries():
    with pytest.raises(SemanticError, match="kind is invalid"):
        _request("invalid")
    with pytest.raises(SemanticError, match="positive integer"):
        _request(CollectiveKind.ALL_GATHER, message_size_bytes=0)
    with pytest.raises(SemanticError, match="whitespace"):
        _request(CollectiveKind.ALL_GATHER, datatype="float 32")
    with pytest.raises(SemanticError, match="boolean"):
        _request(CollectiveKind.ALL_GATHER, inplace="no")
    with pytest.raises(SemanticError, match="must not define reduction_op"):
        _request(CollectiveKind.ALL_GATHER, reduction_op="sum")
    with pytest.raises(SemanticError, match="binary_directory"):
        _request(CollectiveKind.ALL_GATHER, binary_directory=object())
    with pytest.raises(SemanticError, match="NcclTestRequest"):
        build_nccl_tests_command(object())

    request = _request(
        CollectiveKind.ALL_GATHER,
        binary_directory=None,
    )
    assert build_nccl_tests_command(request)[0] == "all_gather_perf"


def test_measurement_run_and_statistics_models_reject_invalid_values():
    measurement = NcclTestMeasurement(1.0, 2.0, 3.0, 0)
    for changes in (
        {"time_us": 0.0},
        {"algorithm_bandwidth_gbps": -1.0},
        {"bus_bandwidth_gbps": math.inf},
        {"wrong_count": -1},
        {"time_us": True},
    ):
        with pytest.raises(SemanticError):
            replace(measurement, **changes)

    out_only = parse_nccl_tests_output(
        "268435456 67108864 float 10.0 100.0 150.0 0",
        268435456,
    )[0]
    assert out_only.in_place is None
    with pytest.raises(SemanticError, match="selector"):
        out_only.selected_time_us(inplace="yes")
    with pytest.raises(SemanticError, match="missing"):
        out_only.selected_time_us(inplace=True)
    for changes in (
        {"metadata_fields": ("",)},
        {"out_of_place": object()},
        {"in_place": object()},
        {"element_count": 0},
    ):
        with pytest.raises(SemanticError):
            replace(out_only, **changes)

    statistics = PerformanceStatistics(
        samples_us=(1.0,),
        sample_count=1,
        median_us=1.0,
        p95_us=1.0,
        mean_us=1.0,
        population_standard_deviation_us=0.0,
        coefficient_of_variation=0.0,
        stable=True,
    )
    with pytest.raises(SemanticError, match="sample count"):
        replace(statistics, sample_count=2)
    with pytest.raises(SemanticError, match="stable"):
        replace(statistics, stable="yes")


def test_parser_and_help_validator_reject_malformed_boundaries():
    for text, message in (
        (
            "268435456 bad float 10.0 100.0 150.0 0",
            "element count",
        ),
        (
            "268435456 1 float bad 100.0 150.0 0",
            "time",
        ),
        (
            "268435456 1 float 10.0 -1.0 150.0 0",
            "bandwidth",
        ),
    ):
        with pytest.raises(SemanticError, match=message):
            parse_nccl_tests_output(text, 268435456)

    with pytest.raises(SemanticError, match="string"):
        parse_nccl_tests_output(object(), 1)
    with pytest.raises(SemanticError, match="expected_bytes"):
        parse_nccl_tests_output("unused", 0)
    with pytest.raises(SemanticError, match="performance row"):
        parse_nccl_tests_output("ignored line\n1 2 3", 1)

    validator = NcclTestsHelpValidator()
    with pytest.raises(SemanticError, match="NcclTestRequest"):
        validator.validate(object(), lambda command: "-b")
    with pytest.raises(SemanticError, match="callable"):
        validator.validate(
            _request(CollectiveKind.ALL_GATHER),
            object(),
        )
    with pytest.raises(SemanticError, match="empty"):
        validator.validate(
            _request(CollectiveKind.ALL_GATHER),
            lambda command: "",
        )
