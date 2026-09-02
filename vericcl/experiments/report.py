from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import io
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Optional, Tuple

from vericcl.artifacts.writer import atomic_write_text
from vericcl.errors import SemanticError
from vericcl.experiments.performance import (
    ActivationEvidence,
    PerformanceResult,
    XmlSource,
)
from vericcl.experiments.state import ExperimentStateStore, TaskStatus
from vericcl.verification.online.model import (
    NcclTestMeasurement,
    NcclTestRun,
)


REQUESTED_SIZE_BYTES = frozenset(
    value * 1024 * 1024 for value in (4, 16, 64, 256)
).union({1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024})

REPORT_FIELDS = (
    "topology",
    "collective",
    "size_bytes",
    "source",
    "xml_name",
    "inplace_algbw_gbps",
    "out_of_place_busbw_gbps",
    "in_place_busbw_gbps",
    "busbw_relative_difference",
    "msccl_activation",
    "wrong_count",
    "selected_k",
    "solver_status",
    "mip_gap",
    "offline_validation",
    "online_validation",
    "tuning_strategy",
    "eligible_for_comparison",
    "baseline_inplace_algbw_gbps",
    "relative_improvement",
)


@dataclass(frozen=True)
class ReportRow:
    topology: str
    collective: str
    size_bytes: int
    source: str
    xml_name: str
    inplace_algbw_gbps: float
    out_of_place_busbw_gbps: float
    in_place_busbw_gbps: float
    busbw_relative_difference: float
    msccl_activation: str
    wrong_count: int
    selected_k: Optional[int]
    solver_status: Optional[str]
    mip_gap: Optional[float]
    offline_validation: str
    online_validation: str
    tuning_strategy: str
    eligible_for_comparison: bool
    baseline_inplace_algbw_gbps: Optional[float]
    relative_improvement: Optional[float]

    def __post_init__(self) -> None:
        for field in (
            "topology",
            "collective",
            "source",
            "xml_name",
            "msccl_activation",
            "offline_validation",
            "online_validation",
            "tuning_strategy",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value or "\x00" in value:
                raise SemanticError("report {} is invalid".format(field))
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 1
        ):
            raise SemanticError("report size_bytes is invalid")
        for field in (
            "inplace_algbw_gbps",
            "out_of_place_busbw_gbps",
            "in_place_busbw_gbps",
            "busbw_relative_difference",
        ):
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise SemanticError("report {} is invalid".format(field))
            object.__setattr__(self, field, float(value))
        if not isinstance(self.eligible_for_comparison, bool):
            raise SemanticError("report comparison eligibility is invalid")


def _row(
    result: PerformanceResult,
    index: int,
    metadata: Mapping[str, object],
) -> ReportRow:
    run = result.runs[index]
    activation = result.activation[index]
    if run.in_place is None:
        raise SemanticError("in-place performance measurement is missing")
    wrong_count = max(
        run.out_of_place.wrong_count,
        run.in_place.wrong_count,
    )
    eligible = activation.confirmed and wrong_count == 0
    return ReportRow(
        topology=result.topology_name,
        collective=result.collective_label,
        size_bytes=run.message_size_bytes,
        source=result.source.value,
        xml_name=result.xml_name,
        inplace_algbw_gbps=run.in_place.algorithm_bandwidth_gbps,
        out_of_place_busbw_gbps=run.out_of_place.bus_bandwidth_gbps,
        in_place_busbw_gbps=run.in_place.bus_bandwidth_gbps,
        busbw_relative_difference=(
            activation.relative_busbw_difference
        ),
        msccl_activation=("confirmed" if activation.confirmed else "unconfirmed"),
        wrong_count=wrong_count,
        selected_k=metadata.get("selected_k"),
        solver_status=metadata.get("solver_status"),
        mip_gap=metadata.get("mip_gap"),
        offline_validation=str(metadata.get("offline_validation", "unknown")),
        online_validation=str(metadata.get("online_validation", "unknown")),
        tuning_strategy=str(metadata.get("tuning_strategy", "none")),
        eligible_for_comparison=eligible,
        baseline_inplace_algbw_gbps=None,
        relative_improvement=None,
    )


def build_report_rows(
    results: Iterable[PerformanceResult],
    *,
    metadata_by_task: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> Tuple[ReportRow, ...]:
    try:
        values = tuple(results)
    except TypeError as error:
        raise SemanticError("performance results must be iterable") from error
    if not all(isinstance(value, PerformanceResult) for value in values):
        raise SemanticError("performance result is invalid")
    metadata = {} if metadata_by_task is None else dict(metadata_by_task)
    rows = []
    for result in values:
        task_metadata = metadata.get(result.task_id, {})
        if not isinstance(task_metadata, Mapping):
            raise SemanticError("report task metadata is invalid")
        for index, run in enumerate(result.runs):
            if run.message_size_bytes not in REQUESTED_SIZE_BYTES:
                continue
            rows.append(_row(result, index, task_metadata))
    best_baseline = {}
    for row in rows:
        if row.source != XmlSource.BASELINE.value or not row.eligible_for_comparison:
            continue
        key = (row.topology, row.collective, row.size_bytes)
        best_baseline[key] = max(
            best_baseline.get(key, 0.0),
            row.inplace_algbw_gbps,
        )
    compared = []
    for row in rows:
        if row.source != XmlSource.VERICCL.value or not row.eligible_for_comparison:
            compared.append(row)
            continue
        baseline = best_baseline.get(
            (row.topology, row.collective, row.size_bytes)
        )
        if baseline is None or baseline <= 0.0:
            compared.append(row)
            continue
        compared.append(
            replace(
                row,
                baseline_inplace_algbw_gbps=baseline,
                relative_improvement=(
                    row.inplace_algbw_gbps - baseline
                )
                / baseline,
            )
        )
    return tuple(
        sorted(
            compared,
            key=lambda row: (
                row.topology,
                row.collective,
                row.size_bytes,
                row.source,
                row.xml_name,
            ),
        )
    )


def _measurement(payload: object) -> NcclTestMeasurement:
    if not isinstance(payload, dict):
        raise SemanticError("measurement placement is invalid")
    try:
        return NcclTestMeasurement(
            time_us=payload["time_us"],
            algorithm_bandwidth_gbps=payload[
                "algorithm_bandwidth_gbps"
            ],
            bus_bandwidth_gbps=payload["bus_bandwidth_gbps"],
            wrong_count=payload["wrong_count"],
        )
    except KeyError as error:
        raise SemanticError("measurement placement field is missing") from error


def _run(payload: object) -> NcclTestRun:
    if not isinstance(payload, dict):
        raise SemanticError("measurement run is invalid")
    try:
        in_place = payload["in_place"]
        metadata_fields = payload["metadata_fields"]
        if not isinstance(metadata_fields, list):
            raise SemanticError("measurement metadata is invalid")
        return NcclTestRun(
            message_size_bytes=payload["message_size_bytes"],
            element_count=payload["element_count"],
            datatype=payload["datatype"],
            metadata_fields=tuple(metadata_fields),
            out_of_place=_measurement(payload["out_of_place"]),
            in_place=None if in_place is None else _measurement(in_place),
        )
    except KeyError as error:
        raise SemanticError("measurement run field is missing") from error


def _activation(payload: object) -> ActivationEvidence:
    if not isinstance(payload, dict):
        raise SemanticError("activation evidence is invalid")
    try:
        return ActivationEvidence(
            info_loaded=payload["info_loaded"],
            relative_busbw_difference=payload[
                "relative_busbw_difference"
            ],
            threshold=payload["threshold"],
            confirmed=payload["confirmed"],
        )
    except KeyError as error:
        raise SemanticError("activation evidence field is missing") from error


def load_performance_results(root: Path) -> Tuple[PerformanceResult, ...]:
    performance_root = Path(root) / "performance"
    results = []
    for measurements_path in sorted(
        performance_root.rglob("measurements.json")
    ):
        activation_path = measurements_path.parent / "activation.json"
        try:
            measurements = json.loads(
                measurements_path.read_text(encoding="utf-8")
            )
            activations = json.loads(
                activation_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, ValueError) as error:
            raise SemanticError("performance result is unreadable") from error
        if (
            not isinstance(measurements, dict)
            or measurements.get("schema_version") != 1
            or not isinstance(activations, dict)
            or activations.get("schema_version") != 1
            or activations.get("task_id") != measurements.get("task_id")
        ):
            raise SemanticError("performance result schema is invalid")
        try:
            runs = tuple(_run(value) for value in measurements["runs"])
            evidence = tuple(
                _activation(value) for value in activations["activation"]
            )
            results.append(
                PerformanceResult(
                    task_id=measurements["task_id"],
                    topology_name=measurements["topology_name"],
                    collective_label=measurements["collective_label"],
                    source=XmlSource(measurements["source"]),
                    xml_name=measurements["xml_name"],
                    runs=runs,
                    activation=evidence,
                    stdout_path=measurements_path.parent / "stdout.log",
                    stderr_path=measurements_path.parent / "stderr.log",
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SemanticError("performance result field is invalid") from error
    return tuple(results)


_OFFLINE_DIMENSIONS = (
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
)


def _solve_metadata(root: Path) -> Mapping[str, Mapping[str, object]]:
    state = ExperimentStateStore(Path(root) / "state.json").load()
    result = {}
    for task_id, record in state.items():
        if (
            task_id.startswith("perf-")
            or task_id.startswith("smoke-")
            or record.status is not TaskStatus.PASSED
            or record.log_path is None
        ):
            continue
        xml_path = Path(record.log_path)
        reports = tuple(xml_path.parent.glob("*_final.validation.json"))
        if len(reports) != 1:
            continue
        try:
            payload = json.loads(reports[0].read_text(encoding="utf-8"))
            validation = payload["validation"]
            metrics = payload["solver_metrics"]
        except (OSError, UnicodeError, ValueError, KeyError, TypeError):
            continue
        offline = "valid" if all(
            validation.get(name, {}).get("status") == "valid"
            for name in _OFFLINE_DIMENSIONS
        ) else "failed"
        online = validation.get("online", {})
        evidence = online.get("evidence", {})
        strategy = payload.get("tuning_strategy", {})
        calibration = evidence.get("calibration") or {}
        result["perf-{}-vericcl".format(task_id)] = {
            "calibration_cache_hits": calibration.get(
                "cache_hit_concurrencies",
                (),
            ),
            "clock_uncertainty_us": evidence.get(
                "trace_clock_uncertainty_us"
            ),
            "mip_gap": metrics.get("mip_gap"),
            "offline_validation": offline,
            "online_validation": online.get("status", "unknown"),
            "proven_optimal": payload.get("proven_optimal", False),
            "selected_k": payload.get("channel_count"),
            "solver_status": metrics.get("status"),
            "tuning_eligible": evidence.get("online_tuning_allowed"),
            "tuning_strategy": json.dumps(
                strategy,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "xml_path": str(xml_path),
        }
    return result


def _markdown(
    rows: Tuple[ReportRow, ...],
    metadata: Mapping[str, Mapping[str, object]],
    failed_tasks: Tuple[str, ...],
) -> str:
    lines = [
        "# V100 VeriCCL performance report",
        "",
        "Primary metric: in-place algorithm bandwidth (GB/s).",
        "Network path: TCP/Ethernet because node2 IB was unavailable and unchanged.",
        (
            "Optimality: best accepted K<=16 candidate; not globally proven "
            "unless proven_optimal=true."
        ),
        "Activation: NCCL INFO evidence plus at least 5% in/out busbw difference.",
        "",
        "## Results",
        "",
        (
            "| Topology | Collective | Size (B) | Source | XML | "
            "In-place algbw (GB/s) | Activation | Relative improvement |"
        ),
        "|---|---:|---:|---|---|---:|---|---:|",
    ]
    for row in rows:
        improvement = (
            ""
            if row.relative_improvement is None
            else "{:.6f}".format(row.relative_improvement)
        )
        lines.append(
            "| {} | {} | {} | {} | {} | {:.6f} | {} | {} |".format(
                row.topology,
                row.collective,
                row.size_bytes,
                row.source,
                row.xml_name,
                row.inplace_algbw_gbps,
                row.msccl_activation,
                improvement,
            )
        )
    lines.extend(("", "## Validation and tuning evidence", ""))
    if metadata:
        for task_id, values in sorted(metadata.items()):
            lines.append(
                (
                    "- {}: xml={}, tuning_strategy={}, "
                    "clock_uncertainty_us={}, tuning_eligible={}, "
                    "calibration_cache_hits={}, proven_optimal={}"
                ).format(
                    task_id,
                    values.get("xml_path"),
                    values.get("tuning_strategy"),
                    values.get("clock_uncertainty_us"),
                    values.get("tuning_eligible"),
                    values.get("calibration_cache_hits"),
                    values.get("proven_optimal"),
                )
            )
    else:
        lines.append("- No accepted VeriCCL solve metadata was found.")
    unconfirmed = tuple(
        "{}:{}:{}:{}".format(
            row.topology,
            row.collective,
            row.size_bytes,
            row.xml_name,
        )
        for row in rows
        if row.msccl_activation != "confirmed"
    )
    lines.extend(("", "## Exclusions", ""))
    lines.append(
        "- Failed tasks: {}".format(
            ", ".join(failed_tasks) if failed_tasks else "none"
        )
    )
    lines.append(
        "- Unconfirmed XML measurements: {}".format(
            ", ".join(unconfirmed) if unconfirmed else "none"
        )
    )
    return "\n".join(lines) + "\n"


def write_report(root: Path) -> Tuple[Path, Path, Path]:
    experiment_root = Path(root).resolve()
    forbidden = Path("/home/cc")
    if experiment_root == forbidden or experiment_root.is_relative_to(forbidden):
        raise SemanticError("report root uses a forbidden path")
    results = load_performance_results(experiment_root)
    metadata = _solve_metadata(experiment_root)
    rows = build_report_rows(results, metadata_by_task=metadata)
    summary = experiment_root / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    json_path = summary / "results.json"
    csv_path = summary / "results.csv"
    markdown_path = summary / "report.md"
    payloads = [asdict(row) for row in rows]
    atomic_write_text(
        json_path,
        json.dumps(
            {"rows": payloads, "schema_version": 1},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=REPORT_FIELDS)
    writer.writeheader()
    writer.writerows(payloads)
    atomic_write_text(csv_path, stream.getvalue())
    state = ExperimentStateStore(experiment_root / "state.json").load()
    failed = tuple(
        task_id
        for task_id, record in sorted(state.items())
        if record.status is TaskStatus.FAILED
    )
    atomic_write_text(markdown_path, _markdown(rows, metadata, failed))
    return csv_path, json_path, markdown_path
