"""Bounded repair and GPU overhead experiments with raw evidence retention."""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib
import json
from pathlib import Path
import statistics

from vericcl.experiments.overhead import (
    PhaseTimer, digest, environment, instrument, json_value,
    input_identity, outcome_summary, registrations, select_backend, write_json,
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


def micro_case(name):
    """Small full-collective regressions; these are not GPU workloads."""
    from vericcl.semantics.collective import CollectiveKind
    from vericcl.topology.model import LaneKey
    from vericcl.verification.bdd_flow import FlowReplacementHint
    from vericcl.verification.flow_index import build_flow_index
    from tests.unit.verification.helpers import inputs, topology
    from tests.unit.xml.helpers import two_rank_allreduce_schedule, two_send_same_lane_schedule

    def retime(transfer, start, end):
        return replace(transfer, st_time=start, ed_time=end, atoms=tuple(
            replace(atom, st_time=start, ed_time=end) for atom in transfer.atoms))

    if name == "allreduce":
        schedule = two_rank_allreduce_schedule()
        schedule = replace(schedule, transfers=(schedule.transfers[0],
                           retime(schedule.transfers[1], 8.0, 9.0)))
        data, transfer_id, path = inputs(), "allreduce-reduce", (1, 0)
    elif name == "broadcast":
        schedule = two_send_same_lane_schedule()
        first, second = schedule.transfers
        atom = second.atoms[0]
        stage = atom.path[0]
        second = replace(second, atoms=(replace(atom, path=(replace(stage,
            symbols=(replace(stage.symbols[0], ready_time=0.0),)),)),))
        schedule = replace(schedule, transfers=(retime(first, 0.0, 3.0),
                                                retime(second, 3.0, 4.0)))
        data = inputs(CollectiveKind.BROADCAST, slices=2)
        transfer_id, path = second.transfer_id, (0, 1)
    else:
        raise ValueError("unknown micro case")
    flow = next(flow for flow in build_flow_index(schedule).flows
                if transfer_id in flow.comparison_transfer_ids)
    hint = FlowReplacementHint(
        source_flow_id=flow.flow_id, demand_id=flow.demand_id,
        candidate_flow_ids=("selected",), candidate_paths={"selected": path},
        candidate_first_lanes={"selected": LaneKey(path[0], path[1], 1)},
        divergence_rank=path[0], waiting_transfer_id=transfer_id,
        bottleneck_lane=flow.lanes[flow.transfer_ids.index(transfer_id)],
        wait_start_us=0.0, wait_end_us=1.0, earliest_candidate_start_us=0.0)
    return schedule, topology(), data, hint


def load_case(args):
    from vericcl.artifacts.writer import load_schedule_sidecar
    from vericcl.input.loader import resolve_inputs
    from vericcl.topology.loader import load_topology
    data = resolve_inputs(args.topology, args.sketch, args.atoms)
    return load_schedule_sidecar(Path(args.schedule)), load_topology(data), data


def run_repair(args):
    from vericcl.solver.budget import ModelBudget
    from vericcl.tuning.model import TuningOverlay
    import vericcl.tuning.impact as impact_module
    import vericcl.tuning.local_milp as local_module
    import vericcl.tuning.repair as greedy_module
    import vericcl.verification.pipeline as pipeline
    if args.micro:
        schedule, topo, data, hint = micro_case(args.micro)
    else:
        schedule, topo, data = load_case(args)
        checked = pipeline.validate_and_lower_candidate(schedule, data, topo)
        if not checked.flow_bdd or not checked.flow_bdd.hints:
            raise ValueError("schedule has no repair hint")
        hint = checked.flow_bdd.hints[args.hint_index]
    refs = registrations()
    overlay = TuningOverlay("overhead-repair", "parent", channel_count=args.channels,
                            milp_parameters=(("Threads", 1), ("Seed", 0)))
    records = []
    payload = {"environment": environment(), "arguments": vars(args),
        "input_identity": input_identity(args, data),
        "parent_sha256": digest(schedule), "parent_completion_time_us": max(
            t.ed_time for t in schedule.transfers), "hint": json_value(hint),
        "transfer_count": len(schedule.transfers), "warmups": 1,
        "records": records, "complete": False}
    write_json(args.output, payload)
    for index in range(args.repeats + 1):
        timer = PhaseTimer()
        with instrument(timer, refs), timer.span("repair_round"):
            closure = impact_module.compute_impact_closure(
                schedule, frozenset({hint.waiting_transfer_id}), topo)
            if args.method == "greedy":
                result = greedy_module.repair_flow_suffix(schedule, hint, overlay, topo, data)
            else:
                result = local_module.solve_local_repair(schedule, hint, closure,
                    overlay, topo, data, ModelBudget(args.timeout, 0.0, args.timeout))
            outcome = (pipeline.validate_and_lower_candidate(result.schedule, data, topo)
                       if result.schedule is not None else None)
        if index:
            records.append({"timing": timer.payload(), "status": result.status.value,
                "evidence": json_value(result.evidence), "impact_count": len(closure.transfer_ids),
                "schedule_sha256": digest(result.schedule) if result.schedule else None,
                "completion_time_us": max(t.ed_time for t in result.schedule.transfers)
                                      if result.schedule else None,
                "validation": outcome_summary(outcome) if outcome else None})
            write_json(args.output, payload, replace=True)
    payload["complete"] = True
    write_json(args.output, payload, replace=True)
    print(json.dumps({"output": args.output, "median_s": statistics.median(
        record["timing"]["wall_time_s"] for record in records),
        "statuses": [record["status"] for record in records]}), flush=True)
    return 0


def online_registrations():
    from vericcl.verification.online.runner import NcclTestsRunner
    refs = list(registrations())
    for module, name, label, priority in (
        ("pipeline", "_preflight", "online_preflight", 25),
        ("pipeline", "_calibrate", "calibration", 25),
        ("runner", "collect_trace_files", "trace_collect", 30),
        ("trace_format", "parse_trace", "trace_parse", 40),
        ("clock_sync", "align_clocks", "trace_align", 40),
        ("trace_analysis", "analyze_trace", "trace_analysis", 40),
    ):
        owner = importlib.import_module("vericcl.verification.online." + module)
        refs.append((getattr(owner, name), label, priority))
    for name, label in (("measure", "release_measurement"),
                        ("validate_release", "release_single"),
                        ("diagnostic", "trace_run"), ("run_auxiliary", "clock_sync")):
        refs.append((getattr(NcclTestsRunner, name), label, 25))
    return tuple(refs)


def online_eligible(outcome):
    from vericcl.artifacts.writer import _offline_valid
    return (outcome.artifact is not None and outcome.artifact.runtime_compatible
            and _offline_valid(outcome))


def online_output_root(args):
    from vericcl.experiments.remote import ExperimentPathPolicy
    from vericcl.experiments.v100 import load_v100_config
    config = load_v100_config(Path(args.config))
    return ExperimentPathPolicy(config.experiment_root).require_allowed(
        Path(args.output).absolute())


def load_calibration_case(args):
    from vericcl.cli.online import representative_calibration_topology
    from vericcl.input.loader import resolve_inputs
    from vericcl.topology.loader import load_topology
    from vericcl.verification.online.calibration import CalibrationRequest
    from vericcl.verification.online.calibration_xml import build_calibration_benchmark
    source = resolve_inputs(args.topology, args.sketch, args.atoms)
    topo = representative_calibration_topology(load_topology(source), "inter_node")
    benchmark = build_calibration_benchmark(CalibrationRequest(
        "inter_node", source.hyperparameters.slice_size_bytes,
        max_calibration_channels=16, datatype="float32"), topo, concurrency=args.channels)
    return benchmark.schedule, topo, benchmark.inputs


def run_online_once(args):
    from vericcl.artifacts.writer import atomic_write_text
    from vericcl.cli.online import build_online_context_factory
    from vericcl.experiments.v100 import (
        load_v100_config, _configured_hosts, _runtime_dependencies, _factory_environment)
    from vericcl.verification.online.pipeline import run_online_validation
    from vericcl.verification.pipeline import validate_and_lower_candidate

    config = load_v100_config(Path(args.config))
    output = online_output_root(args)
    if args.mode == "calibration":
        schedule, topo, data = load_calibration_case(args)
    else:
        schedule, topo, data = load_case(args)
    outcome = validate_and_lower_candidate(schedule, data, topo)
    if not online_eligible(outcome):
        write_json(output / "overhead.json", {
            "environment": environment(), "arguments": vars(args),
            "input_identity": input_identity(args, data),
            "offline": outcome_summary(outcome), "result": {
                "failure_code": "offline_validation_rejected",
                "online_operator_validation": "not_run"}})
        return 1
    _configured_hosts(config)
    output.mkdir(parents=True, exist_ok=True)
    xml_path = output / "schedule.xml"
    atomic_write_text(xml_path, outcome.artifact.xml_text)
    timer = PhaseTimer()
    executor, collector = _runtime_dependencies(config, data.rank_count)
    executor = LoggingExecutor(executor, output / "processes", timer)

    def timed_collector(request):
        with timer.span("trace_fetch_and_collect", 30):
            return collector(request)

    factory = build_online_context_factory(_factory_environment(config, data.rank_count),
                                            executor=executor, trace_collector=timed_collector)
    context = factory(outcome.artifact, schedule, data, xml_path,
                      output / "traces", False, args.timeout, False)
    context = replace(context, single_process_release_validation=not args.full_release)
    with instrument(timer, online_registrations()), timer.span("online_other"):
        result = run_online_validation(context)
    write_json(output / "overhead.json", {"environment": environment(),
        "input_identity": input_identity(args, data),
        "arguments": vars(args), "schedule_sha256": digest(schedule),
        "offline": outcome_summary(outcome), "timing": timer.payload(),
        "result": json_value(result)})
    print(json.dumps({"output": str(output), "wall_s": timer.payload()["wall_time_s"],
        "release": result.release_status.value,
        "trace": result.online_operator_validation.value,
        "failure": result.failure_code}), flush=True)
    return 0 if result.failure_code is None else 1


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("repair", "online", "clock", "calibration"))
    parser.add_argument("--topology")
    parser.add_argument("--sketch")
    parser.add_argument("--atoms")
    parser.add_argument("--schedule")
    parser.add_argument("--config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--micro", choices=("broadcast", "allreduce"))
    parser.add_argument("--method", choices=("greedy", "milp"), default="greedy")
    parser.add_argument("--bdd-backend", choices=("python", "native"), default="native")
    parser.add_argument("--repeats", type=int,
                        help="default: five CPU repeats or one online invocation")
    parser.add_argument("--hint-index", type=int, default=0)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--full-release", action="store_true")
    parser.add_argument("--ranks", type=int, choices=(2, 8, 16), default=8)
    parser.add_argument("--clock-samples", type=int, default=256)
    parser.add_argument("--tcp-only", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats is None:
        args.repeats = 5 if args.mode == "repair" else 1
    if args.repeats < 1 or args.timeout <= 0:
        parser.error("positive repeats and timeout are required")
    if args.mode == "clock" and args.repeats != 1:
        parser.error("clock mode runs once; use distinct output paths for repetitions")
    if args.mode == "calibration" and not all((args.topology, args.sketch, args.atoms)):
        parser.error("calibration requires topology, sketch, and atoms")
    if args.mode in {"repair", "online"} and not args.micro and not all((args.topology, args.sketch, args.atoms, args.schedule)):
        parser.error("all input paths and schedule are required")
    if args.mode in {"online", "clock", "calibration"} and (not args.config or args.micro):
        parser.error("online requires config and a real schedule")
    select_backend(args.bdd_backend)
    if args.mode == "clock":
        return run_clock(args)
    return run_repair(args) if args.mode == "repair" else run_online(args)


if __name__ == "__main__":
    raise SystemExit(main())
