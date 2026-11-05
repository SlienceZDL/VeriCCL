"""Summarize raw overhead records without turning failed runs into successes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from vericcl.experiments.overhead import write_json


def compact_timing(timing):
    total = timing["wall_time_s"]
    values = timing["exclusive_s"]
    if not math.isclose(sum(values.values()), total, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("exclusive timing does not sum to elapsed time")
    return {"wall_time_s": total, "exclusive_s": values,
            "percent": {name: 100 * value / total if total else 0
                        for name, value in values.items()},
            "calls": timing.get("calls", {})}


def compact_evidence(evidence):
    # Long transfer-ID and order-pair lists remain in immutable raw records.
    return {key: value for key, value in evidence.items()
            if not isinstance(value, (list, dict))}


def summarize(data):
    result = {key: data[key] for key in (
        "arguments", "environment", "input_identity", "input_hashes", "warmups",
        "complete", "schedule_sha256", "parent_sha256",
        "parent_completion_time_us", "transfer_count", "attribution_valid",
        "online_acceptance", "purpose", "xml_sha256", "input_sha256") if key in data}
    if "records" in data:
        records = []
        for record in data["records"]:
            item = {key: value for key, value in record.items()
                    if key not in {"timing", "evidence"}}
            item["timing"] = compact_timing(record["timing"])
            if "evidence" in record:
                item["evidence"] = compact_evidence(record["evidence"])
            records.append(item)
        result["records"] = records
        if records:
            values = [record["timing"]["wall_time_s"] for record in records]
            median = statistics.median(values)
            result.update({"samples_s": values, "median_s": median,
                "p95_s": sorted(values)[math.ceil(.95 * len(values)) - 1],
                "median_sample_index": min(range(len(values)),
                    key=lambda index: abs(values[index] - median))})
        return result
    if "timing" in data:
        result["timing"] = compact_timing(data["timing"])
    mode = data.get("arguments", {}).get("mode")
    if mode == "solve":
        result["status"] = ("failed" if "error" in data else
            data.get("result", {}).get("status", "incomplete"))
        result["error"] = data.get("error")
        result["event_errors"] = sorted({event["error"] for event in
            data.get("timing", {}).get("events", []) if event.get("error")})
        workflow = data.get("result", {})
        result["workflow"] = {key: workflow[key] for key in (
            "status", "message", "final_candidate_id", "final_xml", "final_report")
            if key in workflow}
        result["candidate_count"] = len(workflow.get("candidates", []))
    elif "result" in data:
        result["result"] = {key: value for key, value in data["result"].items()
            if key not in {"context_schedule", "trace_analysis", "tuning_evidence"}}
        result["offline"] = data.get("offline")
    elif "returncode" in data:
        result["returncode"] = data["returncode"]
    return result


def build_summary(root):
    root = Path(root).resolve()
    files = sorted({*root.glob("*/replay-*.json"),
                    *root.glob("*/repair-*.json"),
                    *root.glob("*/**/overhead.json")})
    records = {}
    for path in files:
        raw = path.read_bytes()
        record = summarize(json.loads(raw))
        record["raw_sha256"] = hashlib.sha256(raw).hexdigest()
        records[str(path.relative_to(root))] = record
    batches = {}
    for path in sorted(root.glob("*/**/repeats.json")):
        raw = path.read_bytes()
        data = json.loads(raw)
        batches[str(path.relative_to(root))] = {
            "requested": data["requested"], "complete": data["complete"],
            "recorded_count": len(data["records"]),
            "raw_sha256": hashlib.sha256(raw).hexdigest()}
    return {"schema_version": 1, "raw_root": str(root),
        "timing_unit": "seconds", "p95_method": "nearest rank",
        "phase_percent_denominator": "same-record exclusive elapsed time",
        "records": records, "repeat_batches": batches}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    write_json(args.output, build_summary(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
