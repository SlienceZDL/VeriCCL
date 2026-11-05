from dataclasses import replace
from pathlib import Path

import pytest

from vericcl.errors import SemanticError
from vericcl.experiments.performance import (
    ActivationEvidence,
    PerformanceResult,
    XmlSource,
    build_performance_command,
    evaluate_msccl_activation,
    select_baselines,
)
from vericcl.verification.online.model import NcclTestMeasurement, NcclTestRun


def _run(out_busbw, in_busbw):
    return NcclTestRun(
        message_size_bytes=4 * 1024 * 1024,
        element_count=1024 * 1024,
        datatype="float",
        metadata_fields=("none", "-1"),
        out_of_place=NcclTestMeasurement(10.0, 60.0, out_busbw, 0),
        in_place=NcclTestMeasurement(11.0, 65.0, in_busbw, 0),
    )


def test_activation_requires_info_and_five_percent_busbw_difference():
    run = _run(out_busbw=70.0, in_busbw=75.0)

    evidence = evaluate_msccl_activation(
        "NCCL INFO Connected 1 MSCCL algorithms\n",
        run,
    )

    assert evidence.info_loaded is True
    assert evidence.relative_busbw_difference == pytest.approx(5.0 / 75.0)
    assert evidence.confirmed is True


def test_activation_is_unconfirmed_without_info():
    evidence = evaluate_msccl_activation("", _run(70.0, 75.0))

    assert evidence.info_loaded is False
    assert evidence.confirmed is False


def test_activation_requires_both_placements_and_threshold():
    out_only = replace(_run(70.0, 75.0), in_place=None)

    assert evaluate_msccl_activation(
        "NCCL INFO Connected 1 MSCCL algorithms\n",
        out_only,
    ).confirmed is False
    assert evaluate_msccl_activation(
        "NCCL INFO Connected 1 MSCCL algorithms\n",
        _run(70.0, 73.0),
    ).confirmed is False


def test_activation_models_reject_invalid_boundaries():
    evidence = ActivationEvidence(True, 0.1, 0.05, True)

    for changes in (
        {"info_loaded": "yes"},
        {"relative_busbw_difference": -1.0},
        {"threshold": 2.0},
        {"confirmed": "yes"},
    ):
        with pytest.raises(SemanticError):
            replace(evidence, **changes)
    with pytest.raises(SemanticError):
        evaluate_msccl_activation(object(), _run(70.0, 75.0))
    with pytest.raises(SemanticError):
        evaluate_msccl_activation("", object())


def test_vericcl_benchmark_uses_exact_size_and_fifteen_iterations():
    command = build_performance_command(
        binary="/tests/all_gather_perf",
        begin="4M",
        end="4M",
        factor=2,
        iterations=15,
    )

    assert command[-10:] == (
        "-b",
        "4M",
        "-e",
        "4M",
        "-f",
        "2",
        "-g",
        "1",
        "-n",
        "15",
    )


def test_baseline_benchmark_uses_full_range():
    command = build_performance_command(
        binary="/tests/all_reduce_perf",
        begin="4M",
        end="2G",
        factor=2,
        iterations=15,
    )

    assert ("-b", "4M", "-e", "2G", "-f", "2") == command[-10:-4]


def _write_xml(path, *, collective, ranks):
    path.write_text(
        (
            '<algo name="test" nchannels="1" nchunksperloop="1" '
            'proto="Simple" coll="{}" inplace="1" redop="nop" '
            'ngpus="{}" minBytes="1" maxBytes="2"></algo>'
        ).format(collective, ranks),
        encoding="ascii",
    )
    return path


def test_baselines_are_selected_by_xml_contract(tmp_path):
    paths = (
        _write_xml(tmp_path / "ag-16.xml", collective="allgather", ranks=16),
        _write_xml(tmp_path / "ag-8.xml", collective="allgather", ranks=8),
        _write_xml(tmp_path / "ar-16.xml", collective="allreduce", ranks=16),
    )

    selected = select_baselines(
        paths,
        collective="allgather",
        rank_count=16,
    )

    assert tuple(path.name for path in selected) == ("ag-16.xml",)


def test_performance_result_requires_ordered_matching_evidence(tmp_path):
    first = _run(70.0, 75.0)
    second = replace(first, message_size_bytes=16 * 1024 * 1024)
    evidence = evaluate_msccl_activation(
        "NCCL INFO Connected 1 MSCCL algorithms\n",
        first,
    )
    values = {
        "task_id": "task",
        "topology_name": "v100-n2g4",
        "collective_label": "ag",
        "source": XmlSource.VERICCL,
        "xml_name": "schedule.xml",
        "runs": (first, second),
        "activation": (evidence, evidence),
        "stdout_path": Path(tmp_path / "stdout.log"),
        "stderr_path": Path(tmp_path / "stderr.log"),
    }

    PerformanceResult(**values)
    with pytest.raises(SemanticError, match="activation"):
        PerformanceResult(**{**values, "activation": (evidence,)})
    with pytest.raises(SemanticError, match="order"):
        PerformanceResult(**{**values, "runs": (second, first)})
