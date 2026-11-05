"""CPU-only ablation on real two-rank calibration traces, without acceptance."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import statistics

from vericcl.experiments.overhead import PhaseTimer, digest, environment, write_json
from vericcl.experiments.overhead_jobs import load_calibration_case


def verify_xml_identity(path, expected):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
        raise ValueError("trace replay XML identity does not match calibration sidecar")


def select_inplace_records(records, iterations=20):
    invocation_sets = [{record.iteration for record in group}
                       for group in records.values()]
    if not invocation_sets or any(values != invocation_sets[0] for values in invocation_sets):
        raise ValueError("trace invocations differ by rank")
    ids = sorted(invocation_sets[0])
    block = iterations + 1
    if len(ids) not in (block, 2 * block):
        raise ValueError("unexpected trace invocation count")
    offset = block if len(ids) == 2 * block else 0
    selected = frozenset(ids[offset + 1:offset + block])
    return {rank: tuple(record for record in group if record.iteration in selected)
            for rank, group in records.items()}


def run(args):
    from vericcl.verification.online.trace_format import parse_trace
    from vericcl.verification.online.clock_sync import parse_clock_sync_output, align_clocks
    from vericcl.verification.online.trace_analysis import analyze_trace
    from vericcl.verification.pipeline import validate_and_lower_candidate
    from vericcl.xml.trace_sidecar import build_trace_sidecar

    schedule, topo, data = load_calibration_case(args)
    outcome = validate_and_lower_candidate(schedule, data, topo)
    sidecar = build_trace_sidecar(outcome.artifact, schedule)
    root = Path(args.run_directory)
    verify_xml_identity(root / "schedule.xml", outcome.artifact.sha256)
    paths = [root / "traces" / "msccl-step.rank-{}.bin".format(rank) for rank in range(2)]
    clock_path = root / "processes" / "002.stdout.log"
    clock_output = clock_path.read_text()
    records = []
    payload = {"environment": environment(), "arguments": vars(args),
        "online_acceptance": False, "purpose": "component timing only",
        "xml_sha256": outcome.artifact.sha256,
        "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in paths + [clock_path]},
        "warmups": 1, "records": records, "complete": False}
    write_json(args.output, payload)
    for index in range(args.repeats + 1):
        timer = PhaseTimer()
        with timer.span("trace_replay_other"):
            with timer.span("trace_parse", 10):
                by_rank = {rank: parse_trace(path, sidecar) for rank, path in enumerate(paths)}
            raw_count = sum(len(group) for group in by_rank.values())
            with timer.span("trace_select", 10):
                selected = select_inplace_records(by_rank)
            with timer.span("trace_align", 10):
                alignment = align_clocks(selected, parse_clock_sync_output(clock_output))
            uncertainty = max(value.uncertainty_us for value in alignment.transforms.values())
            with timer.span("trace_analysis", 10):
                analysis = analyze_trace(tuple(record for group in selected.values()
                                               for record in group), sidecar, alignment)
        if index:
            records.append({"timing": timer.payload(), "raw_record_count": raw_count,
                "selected_record_count": sum(len(group) for group in selected.values()),
                "clock_uncertainty_us": uncertainty,
                "clock_quality_gate_passed": uncertainty <= 50.0,
                "online_acceptance": False, "analysis_sha256": digest(analysis)})
            write_json(args.output, payload, replace=True)
    payload["complete"] = True
    write_json(args.output, payload, replace=True)
    print({"output": args.output, "median_s": statistics.median(
        record["timing"]["wall_time_s"] for record in records),
        "clock_uncertainty_us": uncertainty, "online_acceptance": False}, flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("topology", "sketch", "atoms", "run-directory", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("positive repeat count required")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
