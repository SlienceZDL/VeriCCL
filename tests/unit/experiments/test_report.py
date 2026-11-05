from pathlib import Path
import json

import pytest

from vericcl.experiments.performance import (
    ActivationEvidence,
    PerformanceResult,
    XmlSource,
)
from vericcl.experiments.report import build_report_rows, write_report
from vericcl.verification.online.model import NcclTestMeasurement, NcclTestRun


def _result(*, source, confirmed, algbw, xml_name="schedule.xml", size=4):
    size_bytes = size * 1024 * 1024
    run = NcclTestRun(
        message_size_bytes=size_bytes,
        element_count=size_bytes // 4,
        datatype="float",
        metadata_fields=("none", "-1"),
        out_of_place=NcclTestMeasurement(100.0, 70.0, 65.0, 0),
        in_place=NcclTestMeasurement(90.0, algbw, 72.0, 0),
    )
    evidence = ActivationEvidence(
        info_loaded=confirmed,
        relative_busbw_difference=(7.0 / 72.0 if confirmed else 0.0),
        threshold=0.05,
        confirmed=confirmed,
    )
    selected = XmlSource(source)
    return PerformanceResult(
        task_id="task-{}-{}".format(source, xml_name),
        topology_name="v100-n2g4",
        collective_label="ag",
        source=selected,
        xml_name=xml_name,
        runs=(run,),
        activation=(evidence,),
        stdout_path=Path("/tmp/stdout.log"),
        stderr_path=Path("/tmp/stderr.log"),
    )


def test_report_excludes_unconfirmed_activation_from_comparison():
    rows = build_report_rows(
        (_result(source="vericcl", confirmed=False, algbw=80.0),)
    )

    assert rows[0].inplace_algbw_gbps == 80.0
    assert rows[0].eligible_for_comparison is False
    assert rows[0].relative_improvement is None


def test_report_compares_vericcl_with_best_confirmed_baseline():
    rows = build_report_rows(
        (
            _result(source="vericcl", confirmed=True, algbw=90.0),
            _result(
                source="baseline",
                confirmed=True,
                algbw=75.0,
                xml_name="baseline-a.xml",
            ),
            _result(
                source="baseline",
                confirmed=True,
                algbw=80.0,
                xml_name="baseline-b.xml",
            ),
        )
    )

    selected = next(row for row in rows if row.source == "vericcl")
    assert selected.baseline_inplace_algbw_gbps == 80.0
    assert selected.relative_improvement == pytest.approx(0.125)


def test_report_filters_unrequested_factor_two_baseline_sizes():
    rows = build_report_rows(
        (
            _result(
                source="baseline",
                confirmed=True,
                algbw=70.0,
                size=8,
            ),
        )
    )

    assert rows == ()


def test_write_report_preserves_primary_metric_and_network_limit(tmp_path):
    directory = tmp_path / "performance" / "vericcl" / "case"
    directory.mkdir(parents=True)
    measurements = {
        "collective_label": "ag",
        "runs": [
            {
                "datatype": "float",
                "element_count": 1024 * 1024,
                "in_place": {
                    "algorithm_bandwidth_gbps": 80.0,
                    "bus_bandwidth_gbps": 72.0,
                    "time_us": 90.0,
                    "wrong_count": 0,
                },
                "message_size_bytes": 4 * 1024 * 1024,
                "metadata_fields": ["none", "-1"],
                "out_of_place": {
                    "algorithm_bandwidth_gbps": 70.0,
                    "bus_bandwidth_gbps": 65.0,
                    "time_us": 100.0,
                    "wrong_count": 0,
                },
            }
        ],
        "schema_version": 1,
        "source": "vericcl",
        "task_id": "perf-case-vericcl",
        "topology_name": "v100-n2g4",
        "xml_name": "schedule.xml",
    }
    activation = {
        "activation": [
            {
                "confirmed": True,
                "info_loaded": True,
                "relative_busbw_difference": 7.0 / 72.0,
                "threshold": 0.05,
            }
        ],
        "schema_version": 1,
        "task_id": "perf-case-vericcl",
    }
    (directory / "measurements.json").write_text(
        json.dumps(measurements),
        encoding="ascii",
    )
    (directory / "activation.json").write_text(
        json.dumps(activation),
        encoding="ascii",
    )

    csv_path, json_path, markdown_path = write_report(tmp_path)

    assert csv_path.is_file()
    assert json_path.is_file()
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Primary metric: in-place algorithm bandwidth (GB/s)." in markdown
    assert "TCP/Ethernet" in markdown
