"""Reproducible wall-clock attribution and immutable-schedule replay."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from enum import Enum
import functools
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import threading
import time
from collections.abc import Mapping


class PhaseTimer:
    def __init__(self, clock=time.perf_counter_ns):
        self.clock = clock
        self.events = []

    @contextmanager
    def span(self, name, priority=0):
        start = self.clock()
        error = None
        try:
            yield
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            end = self.clock()
            self.events.append({
                "name": name, "start_ns": start, "end_ns": end,
                "priority": priority, "thread": threading.get_ident(),
                "error": error,
            })

    def breakdown_ns(self):
        """Partition elapsed time by deepest selected span, never sum nesting.

        Fine solver attribution requires one model worker. Parallel worker
        intervals are retained, but cannot be interpreted as additive costs.
        """
        boundaries = {}
        for index, event in enumerate(self.events):
            boundaries.setdefault(event["start_ns"], []).append((1, index))
            boundaries.setdefault(event["end_ns"], []).append((-1, index))
        active = set()
        result = {}
        previous = None
        for stamp, changes in sorted(boundaries.items()):
            if previous is not None and active:
                selected = max(active, key=lambda i: (
                    self.events[i]["priority"], self.events[i]["start_ns"], -i
                ))
                name = self.events[selected]["name"]
                result[name] = result.get(name, 0) + stamp - previous
            for direction, index in changes:
                if direction == 1:
                    active.add(index)
                else:
                    active.discard(index)
            previous = stamp
        return result

    def payload(self):
        values = self.breakdown_ns()
        total = sum(values.values())
        inclusive = {}
        calls = {}
        for event in self.events:
            name = event["name"]
            inclusive[name] = inclusive.get(name, 0) + (
                event["end_ns"] - event["start_ns"]
            ) / 1e9
            calls[name] = calls.get(name, 0) + 1
        return {
            "wall_time_s": total / 1e9,
            "exclusive_s": {key: value / 1e9 for key, value in values.items()},
            "percent": {key: 100 * value / total if total else 0
                        for key, value in values.items()},
            "inclusive_worker_s": inclusive, "calls": calls,
            "events": self.events,
        }


@contextmanager
def instrument(timer, registrations):
    """Wrap all already-loaded VeriCCL aliases, restoring them on exit."""
    wrappers = {}
    for function, name, priority in registrations:
        @functools.wraps(function)
        def wrapped(*args, _function=function, _name=name,
                    _priority=priority, **kwargs):
            with timer.span(_name, _priority):
                return _function(*args, **kwargs)
        wrappers[id(function)] = (function, wrapped)
    changed = []
    owners = []
    for name, module in tuple(sys.modules.items()):
        if name.startswith("vericcl.") and module is not None:
            owners.append(module)
            owners.extend(value for value in tuple(vars(module).values())
                          if isinstance(value, type)
                          and value.__module__ == name)
    seen = set()
    try:
        for owner in owners:
            if id(owner) in seen:
                continue
            seen.add(id(owner))
            for name, value in tuple(vars(owner).items()):
                pair = wrappers.get(id(value))
                if pair is not None and value is pair[0]:
                    changed.append((owner, name, value))
                    setattr(owner, name, pair[1])
        yield
    finally:
        for owner, name, value in reversed(changed):
            setattr(owner, name, value)


def repeated(operation, repeats=5, warmups=1):
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 0:
        raise ValueError("warmups must be a nonnegative integer")
    for _ in range(warmups):
        operation()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        operation()
        samples.append((time.perf_counter_ns() - started) / 1e9)
    return {"samples_s": samples, "median_s": statistics.median(samples),
            "p95_s": sorted(samples)[math.ceil(.95 * len(samples)) - 1],
            "warmups": warmups}


def json_value(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {field.name: json_value(getattr(value, field.name))
                for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(json_value(item) for item in value)
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def input_identity(args, inputs):
    raw = {}
    for name in ("topology", "sketch", "atoms"):
        path = getattr(args, name, None)
        if path:
            raw[name] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {"normalized_input_sha256": inputs.input_sha256, "raw_sha256": raw}


def write_json(path, value, *, replace=False):
    from vericcl.artifacts.writer import atomic_write_text
    from vericcl.experiments.state import atomic_replace_text
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = atomic_replace_text if replace else atomic_write_text
    writer(path, json.dumps(json_value(value), indent=2, sort_keys=True) + "\n")


def select_backend(name):
    import vericcl.verification.bdd_backend as backend
    module = importlib.import_module("dd.cudd" if name == "native" else "dd.autoref")
    backend.BDD = module.BDD


def registrations():
    # Load consumers before wrapping direct-import aliases.
    import vericcl.workflow
    names = (
        ("input.loader", "resolve_inputs", "input", 10),
        ("topology.loader", "load_topology", "input", 10),
        ("planner.build", "build_plan", "planning", 20),
        ("solver.orchestrator", "solve", "synthesis_other", 10),
        ("solver.routing_milp", "_build_route_model", "model_build", 30),
        ("solver.routing_milp", "_optimize_route_model", "model_optimize", 30),
        ("solver.templates", "build_solver_templates", "template_build", 20),
        ("solver.instantiate", "instantiate_route_patterns", "template_expand", 20),
        ("solver.global_scheduler", "assign_global_resources", "global_schedule", 20),
        ("workflow", "_global_schedule", "composition", 15),
        ("verification.pipeline", "validate_and_lower_candidate", "verify_other", 20),
        ("verification.constraints", "verify_schedule_pre_lowering", "verify_core", 30),
        ("verification.bdd_flow", "analyze_flow_congestion", "bdd_flow", 30),
        ("verification.bdd_order", "analyze_tb_order", "bdd_order", 30),
        ("verification.simulator", "simulate_schedule", "simulation", 30),
        ("verification.flow_index", "build_flow_index", "flow_index", 40),
        ("xml.buffers", "build_buffer_plan", "xml_build", 30),
        ("xml.endpoints", "lower_endpoints", "xml_build", 30),
        ("xml.dependencies", "build_transfer_dag", "xml_build", 30),
        ("xml.list_scheduler", "schedule_threadblocks", "xml_build", 30),
        ("xml.emitter", "emit_xml", "xml_build", 30),
        ("xml.liveness", "verify_buffer_liveness", "verify_buffer", 30),
        ("xml.deadlock", "simulate_endpoint_execution", "verify_deadlock", 30),
        ("xml.parser", "validate_xml", "verify_xml", 30),
        ("xml.compatibility", "check_msccl_compatibility", "verify_runtime", 30),
        ("tuning.engine", "tune", "tuning_other", 10),
        ("tuning.repair", "repair_flow_suffix", "repair_greedy", 20),
        ("tuning.local_milp", "solve_local_repair", "repair_local_model", 15),
        ("tuning.impact", "compute_impact_closure", "repair_impact", 25),
        ("artifacts.writer", "write_candidate_artifact", "artifacts", 15),
        ("artifacts.writer", "write_final_alias", "artifacts", 15),
        ("artifacts.writer", "write_run_summary", "artifacts", 15),
    )
    result = []
    for module, name, label, priority in names:
        owner = importlib.import_module("vericcl." + module)
        function = getattr(owner, name, None)
        if function is not None:
            result.append((function, label, priority))
    return tuple(result)


def environment():
    import dd
    from vericcl.solver.scip_api import ScipAdapter
    root = Path(__file__).resolve().parents[2]
    source_hash = hashlib.sha256()
    for path in sorted((root / "vericcl").rglob("*.py")):
        source_hash.update(str(path.relative_to(root)).encode())
        source_hash.update(b"\0")
        source_hash.update(path.read_bytes())
    return {"hostname": platform.node(), "python": sys.version,
            "platform": platform.platform(), "dd": dd.__version__,
            "scip": ScipAdapter.version(),
            "source_root": str(root), "source_sha256": source_hash.hexdigest()}


def outcome_summary(outcome):
    return {"statuses": {field.name: getattr(outcome.report, field.name).status.value
                         for field in fields(outcome.report)},
            "xml_sha256": outcome.artifact.sha256 if outcome.artifact else None,
            "runtime_compatible": (outcome.artifact.runtime_compatible
                                   if outcome.artifact else None),
            "flow_hints_sha256": digest(outcome.flow_bdd.hints) if outcome.flow_bdd else None,
            "order_hints_sha256": digest(outcome.order_bdd.hints) if outcome.order_bdd else None,
            "flow_evidence": dict(outcome.flow_bdd.evidence) if outcome.flow_bdd else None,
            "completion_time_us": (outcome.simulation.completion_time_us
                                   if outcome.simulation else None)}


def run_solve(args):
    from vericcl.workflow import RunContext, execute_solve
    if args.canonical_states is not None:
        import vericcl.topology.isomorphism as isomorphism
        isomorphism._MAX_CANONICAL_STATES = args.canonical_states
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sketch = json.loads(Path(args.sketch).read_text())
    sketch.setdefault("solver", {}).update({
        "max_parallel_models": args.parallel, "max_threads_per_model": args.threads,
        "max_channels": args.channels, "force_resolve": True,
        "total_solve_timeout_s": args.timeout,
        "per_model_timeout_s": args.model_timeout,
    })
    atoms = json.loads(Path(args.atoms).read_text())
    atoms["strategies"]["hierarchy"] = args.hierarchy
    write_json(output / "sketch.json", sketch)
    write_json(output / "atom.json", atoms)
    timer = PhaseTimer()
    record = {"environment": environment(), "arguments": vars(args),
              "input_hashes": {key: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                               for key, path in (("topology", args.topology),
                                                 ("sketch", output / "sketch.json"),
                                                 ("atom", output / "atom.json"))}}
    try:
        with instrument(timer, registrations()), timer.span("workflow_other"):
            result = execute_solve(RunContext(
                topology_path=Path(args.topology), sketch_path=output / "sketch.json",
                atom_path=output / "atom.json", output_base=output / "run",
                run_id="measured", timeout_s=args.timeout, tune=args.tune,
                environment_signature="overhead-controlled"))
        record["result"] = json_value(result)
    except (Exception, KeyboardInterrupt) as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        record["timing"] = timer.payload()
        record["attribution_valid"] = args.parallel == 1
        write_json(output / "overhead.json", record)
    print(json.dumps({"output": str(output), "wall_s": timer.payload()["wall_time_s"],
                      "error": record.get("error")}), flush=True)
    return 1 if "error" in record else 0


def run_replay(args):
    from vericcl.artifacts.writer import load_schedule_sidecar
    from vericcl.input.loader import resolve_inputs
    from vericcl.topology.loader import load_topology
    import vericcl.verification.pipeline as pipeline
    inputs = resolve_inputs(args.topology, args.sketch, args.atoms)
    topology = load_topology(inputs)
    schedule = load_schedule_sidecar(Path(args.schedule))
    records = []
    payload = {"environment": environment(), "arguments": vars(args),
               "input_identity": input_identity(args, inputs),
               "schedule_sha256": digest(schedule), "warmups": 1,
               "records": records, "complete": False}
    write_json(args.output, payload)
    timer = PhaseTimer()
    refs = registrations()
    pipeline.validate_and_lower_candidate(schedule, inputs, topology)
    for _ in range(args.repeats):
        timer = PhaseTimer()
        with instrument(timer, refs), timer.span("replay_other"):
            outcome = pipeline.validate_and_lower_candidate(schedule, inputs, topology)
        records.append({"timing": timer.payload(), "outcome": outcome_summary(outcome)})
        write_json(args.output, payload, replace=True)
    payload["complete"] = True
    write_json(args.output, payload, replace=True)
    print(json.dumps({"output": args.output, "median_s": statistics.median(
        record["timing"]["wall_time_s"] for record in records)}), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("solve", "replay"))
    parser.add_argument("--topology", required=True)
    parser.add_argument("--sketch", required=True)
    parser.add_argument("--atoms", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--schedule")
    parser.add_argument("--bdd-backend", choices=("python", "native"), default="python")
    parser.add_argument("--repeats", type=int,
                        help="replay sample count; solve is always one invocation")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--model-timeout", type=int, default=10)
    parser.add_argument("--canonical-states", type=int,
                        help="experimental exact-isomorphism search cap; unproved domains remain separate")
    parser.add_argument("--hierarchy", action="store_true")
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats is None:
        args.repeats = 1 if args.mode == "solve" else 5
    if args.mode == "solve" and args.repeats != 1:
        parser.error("solve records one invocation; use separate output directories for repeats")
    if args.repeats < 1 or (args.mode == "replay" and not args.schedule):
        parser.error("positive repeats and replay schedule are required")
    if args.canonical_states is not None and args.canonical_states < 1:
        parser.error("canonical state cap must be positive")
    select_backend(args.bdd_backend)
    return run_solve(args) if args.mode == "solve" else run_replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
