"""Bounded v5 repair micro benchmarks and online overhead runs with raw evidence.

``repair``      micro-benchmarks one v5 repair type on a built-in 3-rank case:
                ``correctness`` (type 1), ``reallocation`` (type 2, cross-path
                slice re-division), ``bottleneck`` (type 3, scheduling
                bottleneck) or ``strategy`` (``solve_repair`` path strategy).
``calibration`` one instrumented v5 ``purpose='calibrate'`` run (GPU probes).
``online``      one instrumented v5 ``purpose='verify'`` run of an admitted
                XML + sidecar with ``trace_and_measure`` enabled (GPU trace).
``clock``       one raw clock-sync launch on the V100 experiment hosts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import statistics
import tempfile
import time

from vericcl.experiments.overhead import (
    PhaseTimer, TimerSwitch, add_request_arguments, bundle_summary, digest, environment,
    has_request, instrument, json_value, measure_primary, parse_overrides, registrations,
    repeated, request_files, write_json,
)


class LoggingExecutor:
    def __init__(self, delegate, directory, timer):
        self.delegate = delegate
        self.directory = Path(directory)
        self.timer = timer
        self.count = 0

    def run(self, request):
        from vericcl.artifacts.writer import atomic_write_text
        if request.timeout_s is not None:
            # Apply the deadline on the execution host, not only to local SSH.
            request = replace(request, command=("/usr/bin/timeout", "--signal=TERM",
                "--kill-after=10", str(request.timeout_s)) + request.command,
                timeout_s=request.timeout_s + 15.0)
        index = self.count
        self.count += 1
        self.directory.mkdir(parents=True, exist_ok=True)
        stem = self.directory / "{:03d}".format(index)
        payload = json_value(request)
        payload["environment"] = {key: value for key, value in request.environment.items()
            if key in {"PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES"}
            or key.startswith(("NCCL_", "MSCCL_", "VERICCL_"))}
        write_json(str(stem) + ".request.json", payload)
        try:
            with self.timer.span("process:" + request.label, 50):
                result = self.delegate.run(request)
        except BaseException as error:
            cause = error
            stdout = stderr = ""
            visited = set()
            while cause is not None and id(cause) not in visited:
                visited.add(id(cause))
                stdout = stdout or getattr(cause, "stdout", None) or ""
                stderr = stderr or getattr(cause, "stderr", None) or ""
                cause = cause.__cause__ or cause.__context__
            def decode(value):
                return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
            atomic_write_text(Path(str(stem) + ".stdout.log"), decode(stdout))
            atomic_write_text(Path(str(stem) + ".stderr.log"), decode(stderr))
            write_json(str(stem) + ".result.json", {
                "error": type(error).__name__, "message": str(error), "returncode": None})
            raise
        atomic_write_text(Path(str(stem) + ".stdout.log"), result.stdout)
        atomic_write_text(Path(str(stem) + ".stderr.log"), result.stderr)
        write_json(str(stem) + ".result.json", {"returncode": result.returncode})
        return result


# ---------------------------------------------------------------------------
# repair micro benchmarks

REPAIR_METHODS = ("correctness", "reallocation", "bottleneck", "strategy")
MICRO_RANKS = 3
SLOW_LINK = (0, 1)


def micro_documents(*, channels=1, model_timeout_s=5.0):
    """Test-only 3-rank broadcast, 4 slices; also the tests/scip repair fixture."""
    ranks = MICRO_RANKS
    return {
        "topology": {"name": "overhead-micro", "ranks": ranks,
                     "nodes": [{"id": 0, "ranks": list(range(ranks))}],
                     "directed_links": [{"src": a, "dst": b, "alpha_us": 1.0, "beta_us_per_byte": 0.01}
                                        for a in range(ranks) for b in range(ranks) if a != b]},
        "sketch": {
            "collective": {"operator": "broadcast"},
            "hyperparameters": {"total_size_bytes": 4096, "slice_size_bytes": 1024},
            "solver": {"per_model_timeout_s": model_timeout_s, "cpu_budget": 2, "max_channels": channels},
            "budgets": {"max_initial_synthesis_s": 10, "max_offline_validation_per_round_s": 5,
                        "max_offline_validation_total_s": 20, "max_repair_per_round_s": 5,
                        "max_repair_total_s": 20, "max_tuning_rounds": 2, "cleanup_reserve_s": 0.5},
            "tolerances": {"absolute": 1e-5, "relative": 1e-5},
            "calibration": {"enabled": False},
        },
        "atom": {"synthesis_method": "constructive_trees"},
        "runtime": {"rank_map": [{"rank": r, "host": "worker-a", "device": r} for r in range(ranks)],
                    "environment": {"NCCL_BUFFSIZE": "16777216"},
                    "scratch_bytes_per_rank": 16777216},
    }


def micro_cost_model(topology):
    """Link 0->1 is ten times slower than every other link."""
    from vericcl.performance.model import CostModel, ServiceCost
    services, resources = {}, {}
    for key in topology.links:
        beta = .1 if (key.src_rank, key.dst_rank) == SLOW_LINK else .01
        suffix = "%d:%d" % (key.src_rank, key.dst_rank)
        services["link:" + suffix] = (ServiceCost(1, beta),)
        services["rrc:" + suffix] = (ServiceCost(1.5, beta),)
        resources["link:" + suffix] = 1 / beta
    for rank in range(MICRO_RANKS):
        services["copy:%d" % rank] = (ServiceCost(.1, .001),)
        services["reduce:%d" % rank] = (ServiceCost(.2, .002),)
    return CostModel("overhead-micro", services, resources)


def micro_operations(*, relayed=None, missing=()):
    """Root 0 sends slices 0..3 to ranks 1 and 2.

    With ``relayed`` (two-flow form) slices in it reach rank 1 through 0->2->1.
    Operations named in ``missing`` are omitted (a correctness violation).
    """
    from vericcl.xml.v5_model import NetworkOperation
    participants = tuple(range(MICRO_RANKS))
    operations = []
    for s in range(4):
        contributors = frozenset({s})
        if relayed is None:
            for dst in (1, 2):
                operations.append(NetworkOperation("slice%d-to%d" % (s, dst), "send", 0, dst, 0, 0,
                    contributors, contributors, participants, frozenset(), frozenset(), float(s)))
            continue
        operations.append(NetworkOperation("s%d-to2" % s, "send", 0, 2, 0, 0, contributors, contributors,
                                           participants, frozenset(), frozenset(), float(s)))
        if s in relayed:
            operations.append(NetworkOperation("s%d-to1" % s, "send", 2, 1, 0, 0, contributors, contributors,
                participants, frozenset({"s%d-to2" % s}), frozenset(), float(s) + .5))
        else:
            operations.append(NetworkOperation("s%d-to1" % s, "send", 0, 1, 0, 0, contributors, contributors,
                participants, frozenset(), frozenset(), float(s)))
    return tuple(op for op in operations if op.operation_id not in missing)


@dataclass
class MicroCase:
    method: str
    description: str
    documents: dict
    lowering: object
    bundle: object
    repair: object
    strategy: object = None
    setup: dict = field(default_factory=dict)


def _hotspots(bundle, lowering):
    from vericcl.symbolic.executor import execute_bundle
    from vericcl.verification.hotspots import locate_hotspots
    execution = execute_bundle(bundle, lowering.inputs, lowering.model)
    return locate_hotspots(execution, bundle, lowering.inputs, lowering.topology, lowering.model)


def micro_case(method, directory, *, channels=1, model_timeout_s=5.0, timeout_s=300.0):
    """Build the incumbent and the repair call; strategy preparation is untimed."""
    from vericcl.execution.budget import StandaloneLease
    from vericcl.input.loader import load_request
    from vericcl.topology import load_topology
    from vericcl.xml.v5_lower import lower_and_check
    from vericcl.xml.v5_model import LoweringContext, NetworkCandidate
    if method not in REPAIR_METHODS:
        raise ValueError("unknown repair micro case")
    documents = micro_documents(channels=channels, model_timeout_s=model_timeout_s)
    directory = Path(directory)
    files = {}
    for name, document in documents.items():
        files[name] = directory / (name + ".json")
        files[name].write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    inputs = load_request(files)
    topology = load_topology(inputs)
    lowering = LoweringContext(inputs, topology, micro_cost_model(topology), StandaloneLease.after(timeout_s))

    def lower(operations, origin):
        candidate = NetworkCandidate(inputs.rank_count, inputs.hyperparameters.slice_count,
                                     inputs.hyperparameters.slice_size_bytes, operations, origin)
        return lower_and_check(candidate, lowering)

    # Repairs are called through their modules at call time, so the
    # instrumented aliases (not names bound here) are timed.
    if method == "correctness":
        from vericcl.tuning import correctness
        bundle = lower(micro_operations(missing=("slice2-to2",)), "overhead-missing-delivery")
        if bundle.can_execute:
            raise ValueError("correctness micro case must start from a violated candidate")
        steps = documents["sketch"].get("repair", {}).get("correctness", {}).get("max_steps", 8)
        return MicroCase(method, "type 1: a missing slice-2 delivery to rank 2 is re-added from an owner",
                         documents, lowering, bundle,
                         lambda context: correctness.repair_correctness(bundle, context.lowering, max_steps=steps))
    if method == "reallocation":
        from vericcl.tuning import allocation
        bundle = lower(micro_operations(relayed=(3,)), "overhead-two-flow")
        report = _hotspots(bundle, lowering)
        records = [record for _, record in allocation.reallocation_strategies(report, bundle, lowering)]
        if not records:
            raise ValueError("reallocation micro case found no hotspot with an idle sibling flow")
        strategy = records[0]
        return MicroCase(method, "type 2: slices move from the hot direct flow to the idle 0->2->1 flow",
                         documents, lowering, bundle,
                         lambda context: allocation.solve_reallocation(bundle, strategy, context), strategy,
                         {"hotspot_count": len(report.hotspots), "strategy_count": len(records)})
    bundle = lower(micro_operations(), "overhead-slow-link")
    if not bundle.can_execute:
        raise ValueError("repair micro case incumbent must be executable")
    if method == "bottleneck":
        from vericcl.tuning import bottleneck
        report = _hotspots(bundle, lowering)
        strategy = next((record for _, record in bottleneck.bottleneck_strategies(report, bundle, lowering)), None)
        if strategy is None:
            raise ValueError("bottleneck micro case found no scheduling bottleneck")
        return MicroCase(method, "type 3: the slow 0->1 hotspot is forwarded by an owner rank (scoped MILP)",
                         documents, lowering, bundle,
                         lambda context: bottleneck.solve_bottleneck(bundle, strategy, report, context), strategy,
                         {"hotspot_count": len(report.hotspots)})
    from vericcl.tuning import strategies
    from vericcl.verification.bottlenecks import StrategyRecord
    strategy = StrategyRecord("path", ("slice0-to1",), paths=(("slice0-to1", (0, 2, 1)),))
    domain = strategies.build_domain(None, strategy)
    return MicroCase(method, "solve_repair: reroute slice 0 to rank 1 through rank 2 (event-model MILP)",
                     documents, lowering, bundle,
                     lambda context: strategies.solve_repair(bundle, domain, strategy, context), strategy)


def repair_record(case, result):
    """Normalize ``(bundle, CorrectnessReport)`` and ``RepairResult`` outcomes."""
    if case.method == "correctness":
        candidate, report = result
        record = {"status": report.status, "model_calls": 0, "attempt_count": len(report.steps),
                  "evidence": report.as_json(), "solver_evidence": []}
    else:
        candidate = result.candidate
        record = {"status": result.status, "model_calls": result.model_calls,
                  "attempt_count": len(result.attempts),
                  "evidence": {"domain": json_value(result.domain), "search_complete": result.search_complete,
                               "acceptance_basis": result.acceptance_basis,
                               "full_search_proven": result.full_search_proven},
                  "solver_evidence": json_value(result.solver_evidence)}
    incumbent = case.bundle.prediction
    record["candidate"] = bundle_summary(candidate)
    record["schedule_sha256"] = None if candidate is None else candidate.xml_sha256
    record["completion_time_us"] = (None if candidate is None or candidate.prediction is None
                                    else candidate.prediction.t_pred_us)
    record["incumbent_t_pred_us"] = None if incumbent is None else incumbent.t_pred_us
    return record


def run_repair(args):
    from vericcl.execution.budget import StandaloneLease
    from vericcl.tuning.strategies import RepairContext
    from vericcl.verification.symbolic_check import clear_check_cache
    with tempfile.TemporaryDirectory(prefix="overhead-micro-") as directory:
        started = time.perf_counter()
        case = micro_case(args.method, directory, channels=args.channels or 1,
                          model_timeout_s=args.model_timeout, timeout_s=args.timeout)
        setup_s = time.perf_counter() - started
    incumbent = case.bundle
    refs = registrations()
    records = []
    payload = {"environment": environment(), "arguments": vars(args), "pipeline": "v5",
        "method": args.method, "case": case.description, "documents": case.documents,
        "input_identity": {"normalized_input_sha256": case.lowering.inputs.input_sha256,
                           "documents_sha256": digest(case.documents)},
        "incumbent": bundle_summary(incumbent), "parent_sha256": incumbent.xml_sha256,
        "parent_completion_time_us": None if incumbent.prediction is None else incumbent.prediction.t_pred_us,
        "transfer_count": len(incumbent.source.operations), "strategy": json_value(case.strategy),
        "setup": dict(case.setup, setup_s=setup_s), "symbolic_cache": "cleared_before_each_sample",
        "warmups": 1, "records": records, "complete": False}
    write_json(args.output, payload)
    switch = TimerSwitch()
    calls = []

    def operation():
        clear_check_cache()
        context = RepairContext(replace(case.lowering, lease=StandaloneLease.after(args.timeout)),
                                candidate_batch_size=4)
        timer = switch.timer = PhaseTimer()
        with timer.span("repair_round"):
            result = case.repair(context)
        calls.append((timer, result))

    # Samples are short: evidence is written after timing, so repeated()'s
    # samples exclude file synchronization. The first call is its warmup.
    with instrument(switch, refs):
        payload["repeated"] = repeated(operation, repeats=args.repeats, warmups=1)
    for timer, result in calls[1:]:
        records.append({"timing": timer.payload(), **repair_record(case, result)})
    payload["complete"] = True
    write_json(args.output, payload, replace=True)
    print(json.dumps({"output": args.output, "median_s": statistics.median(
        record["timing"]["wall_time_s"] for record in records),
        "statuses": [record["status"] for record in records]}), flush=True)
    return 0


# ---------------------------------------------------------------------------
# online (GPU) overhead: v5 calibration and traced verification runs

def online_output_root(args):
    """Absolute output root; with ``--config`` it must lie in the experiment root."""
    path = Path(args.output).absolute()
    if getattr(args, "config", None):
        from vericcl.experiments.remote import ExperimentPathPolicy
        from vericcl.experiments.v100 import load_v100_config
        config = load_v100_config(Path(args.config))
        path = ExperimentPathPolicy(config.experiment_root).require_allowed(path)
    return path


def online_overrides(args):
    overrides = parse_overrides(getattr(args, "set", None))
    if args.mode == "online":
        overrides.setdefault("sketch.trace_and_measure.enabled", True)
    return overrides


def run_online_once(args, *, services=None):
    output = online_output_root(args)
    output.mkdir(parents=True, exist_ok=True)
    record = {"environment": environment(), "arguments": vars(args),
              "online_acceptance": args.mode == "online"}
    try:
        files = request_files(args, output / "request")
        overrides = online_overrides(args)
    except (OSError, ValueError) as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
        record["timing"] = PhaseTimer().payload()
    else:
        measure_primary(files, overrides, output,
                        purpose="calibrate" if args.mode == "calibration" else "verify",
                        refs=registrations(), xml_path=getattr(args, "xml", None),
                        sidecar_path=getattr(args, "sidecar", None),
                        cost_model_path=getattr(args, "model", None), services=services, record=record)
    write_json(output / "overhead.json", record)
    result = record.get("result", {})
    print(json.dumps({"output": str(output), "wall_s": record["timing"]["wall_time_s"],
        "execution": result.get("execution"), "stop_reason": result.get("stop_reason"),
        "exit_code": record.get("exit_code"), "phases": record.get("phases"),
        "error": record.get("error")}), flush=True)
    return 0 if "error" not in record and record.get("exit_code") == 0 else 1


def run_online(args):
    if args.repeats == 1:
        return run_online_once(args)
    records = []
    root = online_output_root(args)
    write_json(root / "repeats.json", {"requested": args.repeats,
                                      "records": [], "complete": False})
    code = 0
    for index in range(args.repeats):
        child = argparse.Namespace(**vars(args))
        child.output = str(root / "repeat-{:03d}".format(index))
        child.repeats = 1
        code = max(code, run_online_once(child))
        records.append(json.loads((Path(child.output) / "overhead.json").read_text()))
        write_json(root / "repeats.json", {"requested": args.repeats,
            "records": records, "complete": len(records) == args.repeats}, replace=True)
    return code


def run_clock(args):
    from vericcl.experiments.remote import ExperimentPathPolicy
    from vericcl.experiments.v100 import (
        load_v100_config, _configured_hosts, _runtime_dependencies, build_mpi_prefix)
    from vericcl.verification.online.runner import ProcessRequest
    config = load_v100_config(Path(args.config))
    root = ExperimentPathPolicy(config.experiment_root).require_allowed(Path(args.output).absolute())
    _configured_hosts(config)
    executor, _ = _runtime_dependencies(config, args.ranks)
    timer = PhaseTimer()
    runner = LoggingExecutor(executor, root / "processes", timer)
    import os
    env = {"PATH": str(config.mpi_launcher.parent) + ':' + config.environment.get('PATH', os.environ.get('PATH', '/usr/bin:/bin')),
           "LD_LIBRARY_PATH": config.environment.get('LD_LIBRARY_PATH', os.environ.get('LD_LIBRARY_PATH', ''))}
    prefix = build_mpi_prefix(config, args.ranks, tuple(env))
    if args.tcp_only:
        prefix = tuple("tcp,self" if item == "tcp,self,vader" else item for item in prefix)
    request = ProcessRequest(prefix + (str(config.clock_sync_binary), str(args.clock_samples)),
                             env, "clock-only", timeout_s=args.timeout)
    with timer.span("clock_total"):
        result = runner.run(request)
    write_json(root / "overhead.json", {"arguments": vars(args), "environment": environment(),
        "timing": timer.payload(), "returncode": result.returncode})
    print(json.dumps({"output": str(root), "returncode": result.returncode,
                      "wall_s": timer.payload()["wall_time_s"]}), flush=True)
    return result.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("repair", "online", "clock", "calibration"))
    add_request_arguments(parser)
    parser.add_argument("--xml", help="online: admitted v5 schedule.xml")
    parser.add_argument("--sidecar", help="online: its v5 sidecar.json")
    parser.add_argument("--model", help="online/calibration: starting cost model JSON")
    parser.add_argument("--config", help="clock: V100 experiment config; online: output-root policy")
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=REPAIR_METHODS, help="repair: micro case and repair type")
    parser.add_argument("--repeats", type=int,
                        help="default: five CPU repeats or one online invocation")
    parser.add_argument("--channels", type=int, help="repair: micro case max_channels (default 1)")
    parser.add_argument("--timeout", type=float, default=300,
                        help="repair lease per sample; clock process timeout")
    parser.add_argument("--model-timeout", type=float, default=5.0,
                        help="repair: micro case sketch.solver.per_model_timeout_s")
    parser.add_argument("--ranks", type=int, choices=(2, 8, 16), default=8)
    parser.add_argument("--clock-samples", type=int, default=256)
    parser.add_argument("--tcp-only", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats is None:
        args.repeats = 5 if args.mode == "repair" else 1
    if args.repeats < 1 or args.timeout <= 0 or args.model_timeout <= 0:
        parser.error("positive repeats and timeouts are required")
    if args.mode == "clock" and (args.repeats != 1 or not args.config):
        parser.error("clock mode runs once with --config; use distinct output paths for repetitions")
    if args.mode == "repair" and (not args.method or has_request(args) or args.xml or args.sidecar):
        parser.error("repair requires --method and uses only its built-in micro case")
    if args.mode in {"online", "calibration"} and not has_request(args):
        parser.error("online and calibration require the four v5 documents or --effective-request")
    if args.mode == "online" and not (args.xml and args.sidecar):
        parser.error("online requires an admitted --xml and --sidecar")
    if args.mode == "calibration" and (args.xml or args.sidecar):
        parser.error("calibration measures components, not a supplied schedule")
    if args.mode == "clock":
        return run_clock(args)
    return run_repair(args) if args.mode == "repair" else run_online(args)


if __name__ == "__main__":
    raise SystemExit(main())
