"""CPU-only replay of one retained v5 native trace capture, without acceptance.

The input is a schedule launch directory written by the v5 native driver
(``<run>/jobs/schedule-<id>/`` with ``schedule.xml``, ``launch-result.json`` and
``collected/<host>/native/{trace,clock}.rank-<r>.*``) plus the admitted
candidate's sidecar. Each sample re-runs ``capture_once`` on the retained raw
files (parse, clock alignment, endpoint pairing) and ``analyze_trace_causality``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from vericcl.experiments.overhead import (
    PhaseTimer, TimerSwitch, add_request_arguments, digest, environment, file_sha256, has_request,
    input_identity, instrument, parse_overrides, registrations, request_files, write_json,
)


def verify_xml_identity(path, expected):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
        raise ValueError("trace replay XML identity does not match the candidate sidecar")


def capture_files(launch_directory, rank_map):
    """Retained per-rank trace and clock files, as live.native's trace driver reads them."""
    root = Path(launch_directory)
    traces, clocks = {}, {}
    for mapping in rank_map:
        rank, host = mapping["rank"], mapping["host"]
        base = root / "collected" / host / "native"
        if (base / ("trace.rank-%d.bin" % rank)).is_file():
            traces[rank] = base / ("trace.rank-%d.bin" % rank)
        if (base / ("clock.rank-%d.log" % rank)).is_file():
            clocks[rank] = base / ("clock.rank-%d.log" % rank)
    return traces, clocks


def load_raw_capture(launch_directory, rank_map):
    """Rebuild the RawTraceCapture of one native schedule launch from its evidence."""
    from vericcl.verification.online.clock_sync import parse_clock_sync_output
    from vericcl.verification.online.trace_once import RawTraceCapture
    root = Path(launch_directory)
    evidence = json.loads((root / "launch-result.json").read_text())
    results = evidence.get("rank_results") or []
    traces, clocks = capture_files(root, rank_map)
    samples = []
    for rank in sorted(clocks):
        samples.extend(parse_clock_sync_output(clocks[rank].read_text()))
    metadata = {mapping["rank"]: next((item for item in results if item.get("rank") == mapping["rank"]), {})
                for mapping in rank_map}
    failure = evidence.get("error") or (None if evidence.get("gpu_released") and evidence.get("cleanup_confirmed")
                                        else "native ownership release is incomplete")
    issues = () if not failure else ({"rank": None, "reason": failure, "blocks_acceptance": True},)
    return RawTraceCapture(traces, tuple(samples), metadata, "native_msccl", issues)


class ReplayDriver:
    """Returns the retained capture instead of launching (``capture_once`` driver)."""
    def __init__(self, raw):
        self.raw = raw

    def capture_once(self, plan):
        return self.raw


def warmup_iterations(raw, default):
    values = {item.get("warmup_iterations") for item in raw.metadata_by_rank.values() if item}
    values.discard(None)
    if len(values) > 1:
        raise ValueError("trace ranks report different warmup iteration counts")
    return values.pop() if values else default


def run(args):
    from vericcl.errors import SemanticError
    from vericcl.execution.budget import StandaloneLease
    from vericcl.input.loader import load_request
    from vericcl.performance.initial import initial_cost_model
    from vericcl.performance.serialization import load_model
    from vericcl.planner.parameters import bind_artifact_partition
    from vericcl.topology import load_topology
    from vericcl.verification.online import trace_causality, trace_once
    from vericcl.xml.trace_sidecar import build_v5_trace_sidecar
    from vericcl.xml.v5_lower import check_xml_bundle
    from vericcl.xml.v5_model import LoweringContext

    launch = Path(args.launch_directory)
    output = Path(args.output)
    files = request_files(args, output.parent / (output.stem + "-request"))
    inputs = load_request(files, parse_overrides(args.set), override_input=True, purpose="verify")
    xml_path = Path(args.xml) if args.xml else launch / "schedule.xml"
    sidecar = json.loads(Path(args.sidecar).read_text())
    inputs = bind_artifact_partition(inputs, sidecar)
    model = load_model(Path(args.model)) if args.model else initial_cost_model(inputs)
    bundle = check_xml_bundle(xml_path.read_text(), sidecar, LoweringContext(
        inputs, load_topology(inputs), model, StandaloneLease.after(args.timeout)))
    if not bundle.can_execute:
        raise ValueError("trace replay requires an admitted candidate")
    verify_xml_identity(launch / "schedule.xml", bundle.xml_sha256)
    trace_sidecar = build_v5_trace_sidecar(bundle)
    rank_map = inputs.request_config["runtime"]["rank_map"]
    raw = load_raw_capture(launch, rank_map)
    warmup = warmup_iterations(raw, inputs.request_config["sketch"]["trace_and_measure"]["warmup_iterations"])
    traces, clocks = capture_files(launch, rank_map)
    evidence = [launch / "launch-result.json", *traces.values(), *clocks.values()]
    ranks = tuple(sorted(mapping["rank"] for mapping in rank_map))
    refs = registrations()
    records = []
    payload = {"environment": environment(), "arguments": vars(args), "pipeline": "v5",
        "online_acceptance": False, "purpose": "component timing only",
        "input_identity": input_identity(args, inputs), "xml_sha256": bundle.xml_sha256,
        "input_sha256": {str(path): file_sha256(path) for path in evidence},
        "trace_ranks": sorted(traces), "clock_sample_count": len(raw.clock_samples),
        "warmup_iterations": warmup, "warmups": 1, "records": records, "complete": False}
    write_json(args.output, payload)
    switch = TimerSwitch()
    with instrument(switch, refs):
        for index in range(args.repeats + 1):
            timer = switch.timer = PhaseTimer()
            causality = error = None
            with timer.span("trace_replay_other"):
                lease = StandaloneLease.after(args.timeout)
                # Module lookups at call time use the instrumented aliases.
                trace = trace_once.capture_once(trace_sidecar, trace_once.TraceContext(
                    True, ranks, ReplayDriver(raw), lease=lease, warmup_iterations=warmup))
                try:
                    causality = trace_causality.analyze_trace_causality(bundle, trace, lease=lease)
                except SemanticError as caught:
                    error = str(caught)
            if not index:
                continue
            transforms = {} if trace.alignment is None else trace.alignment.transforms
            uncertainty = max((value.uncertainty_us for value in transforms.values()), default=None)
            records.append({"timing": timer.payload(), "trace_status": trace.status,
                "global_complete": trace.global_complete,
                "record_count": sum(len(group) for group in trace.records_by_rank.values()),
                "aligned_ranks": list(trace.aligned_ranks), "issue_count": len(trace.issues),
                "uncertain_relation_count": len(trace.uncertain_relations),
                "global_span_us": trace.global_span_us, "clock_uncertainty_us": uncertainty,
                "online_acceptance": False, "causality_error": error,
                "causality_sha256": None if causality is None else digest(causality)})
            write_json(args.output, payload, replace=True)
    payload["complete"] = True
    write_json(args.output, payload, replace=True)
    print(json.dumps({"output": args.output, "median_s": statistics.median(
        record["timing"]["wall_time_s"] for record in records),
        "trace_status": records[-1]["trace_status"], "online_acceptance": False}), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_request_arguments(parser)
    parser.add_argument("--launch-directory", required=True,
                        help="native schedule launch directory (jobs/schedule-<id>) of a v5 run")
    parser.add_argument("--sidecar", required=True, help="the launched candidate's v5 sidecar.json")
    parser.add_argument("--xml", help="candidate XML (default: the launch directory's schedule.xml)")
    parser.add_argument("--model", help="cost model JSON used for the candidate check")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=300, help="lease seconds per sample")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.timeout <= 0:
        parser.error("positive repeat count and timeout required")
    if not has_request(args):
        parser.error("the four v5 documents or --effective-request are required")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
