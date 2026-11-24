"""Reproducible wall-clock attribution of the v5 pipeline and immutable-XML replay.

``solve`` runs one v5 primary workflow (initial synthesis, offline verification
and, with ``--tune``, bounded analysis/repair rounds) with function-level spans
and retains the run's own ``reports/timing.json`` phase totals. ``replay``
re-checks one admitted v5 XML + sidecar with ``check_xml_bundle`` several
times (lowering checks, symbolic BDD checks and prediction; no synthesis).
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import fields, is_dataclass
from enum import Enum
from fractions import Fraction
import functools
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import threading
import time
import traceback
from collections.abc import Mapping


REQUEST_DOCUMENTS = ("topology", "sketch", "atom", "runtime")
# Iterator protocol exits are control flow, not failed phases.
_CONTROL_FLOW = (StopIteration, StopAsyncIteration, GeneratorExit)


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
        except _CONTROL_FLOW:
            raise
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


class TimerSwitch:
    """Instrumentation target whose PhaseTimer is replaced per sample.

    Installing instrumentation scans every loaded module; repeated samples
    install it once and switch timers, so samples exclude that setup cost.
    """
    def __init__(self, timer=None):
        self.timer = timer

    def span(self, name, priority=0):
        if self.timer is None:
            return nullcontext()
        return self.timer.span(name, priority)


def _timed_stream(timer, name, priority, stream):
    """Attribute every resumption of a lazy candidate stream, not its creation."""
    try:
        while True:
            with timer.span(name, priority):
                try:
                    item = next(stream)
                except StopIteration as stop:
                    return stop.value
            yield item
    finally:
        with timer.span(name, priority):
            stream.close()


def _owners():
    owners, seen = [], set()
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("vericcl.") or module is None:
            continue
        for owner in [module] + [value for value in tuple(vars(module).values())
                                 if isinstance(value, type) and value.__module__ == name]:
            if id(owner) not in seen:
                seen.add(id(owner))
                owners.append(owner)
    return owners


def _swap(table):
    """Replace every loaded alias whose identity is in ``table`` (old -> new)."""
    changed = []
    for owner in _owners():
        for name, value in tuple(vars(owner).items()):
            if isinstance(value, (classmethod, staticmethod)):
                pair = table.get(id(value.__func__))
                if pair is not None and value.__func__ is pair[0]:
                    changed.append((owner, name, value))
                    setattr(owner, name, type(value)(pair[1]))
                continue
            pair = table.get(id(value))
            if pair is not None and value is pair[0]:
                changed.append((owner, name, value))
                setattr(owner, name, pair[1])
    return changed


@contextmanager
def instrument(timer, registrations):
    """Wrap all already-loaded VeriCCL aliases, restoring them on exit.

    Direct imports (``from x import f``) are separate module attributes with
    the same identity, so every loaded alias is patched. Class, static and
    class methods are patched on their defining class. A returned generator is
    timed per resumption. Aliases created by modules first imported while
    instrumented are reverted on exit as well.
    """
    table = {}
    for function, name, priority in registrations:
        def wrapped(*args, _function=function, _name=name,
                    _priority=priority, **kwargs):
            with timer.span(_name, _priority):
                result = _function(*args, **kwargs)
            if inspect.isgenerator(result):
                return _timed_stream(timer, _name, _priority, result)
            return result
        functools.update_wrapper(wrapped, function)
        table[id(function)] = (function, wrapped)
    changed = _swap(table)
    try:
        yield
    finally:
        for owner, name, value in reversed(changed):
            setattr(owner, name, value)
        _swap({id(wrapper): (wrapper, function) for function, wrapper in table.values()})


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
    if isinstance(value, Fraction):
        return [value.numerator, value.denominator]
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: json_value(getattr(value, field.name))
                for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((json_value(item) for item in value), key=repr)
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def digest(value):
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_identity(args, inputs):
    raw = {}
    for name in REQUEST_DOCUMENTS + ("effective_request",):
        path = getattr(args, name, None)
        if path:
            raw[name] = file_sha256(path)
    return {"normalized_input_sha256": inputs.input_sha256, "raw_sha256": raw}


def write_json(path, value, *, replace=False):
    from vericcl.artifacts.writer import atomic_write_text
    from vericcl.experiments.state import atomic_replace_text
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = atomic_replace_text if replace else atomic_write_text
    writer(path, json.dumps(json_value(value), indent=2, sort_keys=True) + "\n")


def parse_overrides(values):
    """``FIELD=JSON`` items, as accepted by the v5 CLI ``--set`` option."""
    result = {}
    for item in values or ():
        if "=" not in item:
            raise ValueError("--set requires FIELD=JSON")
        path, text = item.split("=", 1)
        if path in result:
            raise ValueError("duplicate --set field: " + path)
        result[path] = json.loads(text)
    return result


def request_files(args, scratch):
    """The four v5 request documents; a run's effective-request.json is split first."""
    effective = getattr(args, "effective_request", None)
    if effective:
        config = json.loads(Path(effective).read_text())
        missing = [name for name in REQUEST_DOCUMENTS if not isinstance(config.get(name), dict)]
        if missing:
            raise ValueError("effective request lacks " + ", ".join(missing))
        scratch = Path(scratch)
        scratch.mkdir(parents=True, exist_ok=True)
        result = {}
        for name in REQUEST_DOCUMENTS:
            path = scratch / (name + ".json")
            path.write_text(json.dumps(config[name], indent=2, sort_keys=True) + "\n")
            result[name] = path.resolve()
        return result
    missing = [name for name in REQUEST_DOCUMENTS if not getattr(args, name, None)]
    if missing:
        raise ValueError("v5 request requires --" + ", --".join(missing))
    return {name: Path(getattr(args, name)).resolve() for name in REQUEST_DOCUMENTS}


# (module below vericcl, attribute path, label, priority). The deepest
# (highest-priority) active span owns elapsed time; containers are low,
# leaves high. Labels are the reported phases.
PHASES = (
    ("input.schema", "resolve_request", "input", 10),
    ("topology.loader", "load_topology", "input", 10),
    ("planner.parameters", "initial_parameter_domain", "planning", 20),
    ("planner.parameters", "resolve_parameters", "planning", 20),
    ("planner.group_search", "enumerate_hierarchy_plans", "planning", 20),
    ("solver.history_index", "find_schedules", "history", 20),
    # Initial synthesis: both methods return lazy candidate streams.
    ("planner.group_search", "plan_hierarchy", "synthesis", 15),
    ("planner.group_search", "CandidateStream.__iter__", "synthesis", 15),
    ("solver.trees.operators", "construct_routes", "synthesis", 15),
    ("solver.trees.timing", "CandidateStream.__iter__", "synthesis", 15),
    ("composer.compose", "compose_routes", "composition", 25),
    ("solver.trees.operators", "RouteSearch.solve", "model_build", 25),
    ("solver.trees.timing", "solve_fixed_seed", "model_build", 25),
    ("solver.trees.timing", "solve_fixed_routes", "model_build", 25),
    ("solver.routing_milp", "solve_route_milp", "model_build", 25),
    ("solver.hierarchy_fine", "solve_fine_pattern", "model_build", 25),
    ("solver.milp", "solve_milp", "model_build", 25),
    ("solver.scip_api", "ScipAdapter.solve", "model_optimize", 40),
    # Lowering and offline verification of each candidate.
    ("xml.v5_lower", "lower_and_check", "verify_other", 25),
    ("xml.v5_lower", "evaluate_program", "verify_other", 25),
    ("xml.v5_lower", "check_xml_bundle", "verify_other", 25),
    ("xml.v5_buffers", "build_storage_plan", "lowering", 30),
    ("xml.v5_program", "lower_program", "lowering", 30),
    ("xml.v5_program", "emit_native_xml", "lowering", 30),
    ("verification.symbolic_check", "check_bundle", "symbolic_check", 30),
    ("symbolic.executor", "execute_bundle", "symbolic_execute", 35),
    ("performance.simulation", "predict", "prediction", 30),
    # Tuning-round analysis.
    ("verification.global_bound", "certify_global_cut", "global_bound", 20),
    ("verification.global_bound", "check_global_cut", "global_bound", 20),
    ("verification.bottlenecks", "analyze", "analysis", 15),
    ("verification.hotspots", "locate_hotspots", "hotspots", 30),
    # Repair types 1-3, strategy repair and resynthesis.
    ("tuning.correctness", "repair_correctness", "repair_correctness", 15),
    ("tuning.allocation", "solve_reallocation", "repair_reallocation", 15),
    ("tuning.bottleneck", "solve_bottleneck", "repair_bottleneck", 15),
    ("tuning.strategies", "solve_repair", "repair_strategy", 15),
    ("tuning.resynthesis", "ResynthesisStream.__next__", "repair_resynthesis", 15),
    ("tuning.event_columns", "solve_event_columns", "event_columns", 20),
    # Online calibration, trace capture and measured acceptance.
    ("verification.online.campaign", "run_calibration", "calibration", 15),
    ("verification.online.probe_jobs", "plan_probes", "calibration_plan", 20),
    ("verification.online.probe_jobs", "execute_probe", "probe_job", 20),
    ("verification.online.probe_xml", "build_probe_xml", "probe_xml", 30),
    ("verification.online.remote", "NativeProbeDriver._launch", "probe_launch", 25),
    ("verification.online.remote", "NativeProbeDriver._collect_evidence", "probe_collect", 30),
    ("verification.online.component_fitting", "fit_components", "calibration_fit", 20),
    ("verification.online.model_publication", "publish_model", "model_publish", 20),
    ("verification.online.component_cache", "lookup_calibration", "calibration_cache", 30),
    ("verification.online.component_cache", "save_calibration", "calibration_cache", 30),
    ("verification.online.trace_once", "capture_once", "trace_capture", 20),
    ("verification.online.trace_format", "parse_partial_trace", "trace_parse", 30),
    ("verification.online.clock_sync", "align_bracketed_globaltimers", "trace_align", 30),
    ("verification.online.trace_causality", "analyze_trace_causality", "trace_causality", 20),
    ("verification.online.acceptance", "compare_measured", "measured_acceptance", 20),
    ("execution.processes", "run_owned_process", "process", 50),
    # Durable evidence (snapshots, candidates, checkpoints, events).
    ("execution.context", "RunContext.create", "artifacts", 20),
    ("execution.context", "RunContext.write_bytes", "artifacts", 45),
    ("execution.context", "finalize_run", "artifacts", 20),
    ("execution.events", "emit_event", "artifacts", 45),
    ("workflow_state", "persist_candidate", "artifacts", 20),
    ("workflow_state", "save_checkpoint", "artifacts", 20),
)
# Imported lazily by the workflow; optional on hosts without the native runtime.
OPTIONAL_PHASES = (
    ("live.native", "NativeScheduleDriver._launch", "schedule_launch", 25),
    ("live.native", "NativeScheduleDriver.measure", "measured_launch", 20),
)
# Consumers that bind phase functions by direct import; load before wrapping.
_CONSUMERS = ("workflow_v5", "workflow_state", "xml.lower", "xml.parser", "solver.constructive",
              "tuning.proposals", "xml.trace_sidecar", "solver.worker_scope")


def _resolve(owner, path):
    value = owner
    for part in path.split("."):
        value = inspect.getattr_static(value, part)
    if isinstance(value, (classmethod, staticmethod)):
        value = value.__func__
    if not callable(value):
        raise AttributeError(path + " is not callable")
    return value


def registrations(strict=False):
    """Resolve PHASES to ``(function, label, priority)``; strict reports drift."""
    for name in _CONSUMERS:
        importlib.import_module("vericcl." + name)
    result, seen, missing = [], set(), []
    for table, required in ((PHASES, True), (OPTIONAL_PHASES, False)):
        for module, path, label, priority in table:
            try:
                function = _resolve(importlib.import_module("vericcl." + module), path)
            except (ImportError, AttributeError) as error:
                if required:
                    missing.append("{}:{} ({})".format(module, path, error))
                continue
            if id(function) not in seen:
                seen.add(id(function))
                result.append((function, label, priority))
    if strict and missing:
        raise LookupError("overhead phases no longer resolve: " + "; ".join(missing))
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


def bundle_summary(bundle):
    """Compact identity and verdicts of one v5 CandidateBundle."""
    if bundle is None:
        return None
    prediction = bundle.prediction
    checks = dict(bundle.checks)
    return {"can_execute": bool(bundle.can_execute), "xml_sha256": bundle.xml_sha256,
            "checks": {key: check.status.value for key, check in checks.items()},
            "invalid_messages": {key: check.message for key, check in checks.items()
                                 if check.status.value == "invalid"},
            "t_pred_us": None if prediction is None else prediction.t_pred_us,
            "model_version": None if prediction is None else prediction.model_version,
            "model_coverage": None if prediction is None else prediction.model_coverage,
            "operation_count": (None if bundle.executable is None
                                else len(bundle.executable.operations))}


def execute_primary(request, context, *, services=None):
    """``run_primary`` + ``finalize_run`` as ``vericcl.cli.lifecycle`` does, silently.

    Returns ``(RunResult, FinalizationResult, exception or None)``.
    """
    from vericcl.errors import SolverUnavailableError
    from vericcl.execution.context import StorageFailure, finalize_run
    from vericcl.execution.events import emit_event
    from vericcl.execution.status import RunResult
    from vericcl.workflow_v5 import run_primary
    error = None
    try:
        result = run_primary(request, context, services=services)
    except StorageFailure:
        raise
    except BaseException as caught:
        error = caught
        if isinstance(caught, KeyboardInterrupt):
            category, execution = "cancelled", "cancelled"
        elif isinstance(caught, SolverUnavailableError):
            category, execution = "backend", "failed"
        else:
            category, execution = "internal", "failed"
        emit_event(context, "execution_exception", {"message": str(caught),
                                                    "traceback": traceback.format_exc()})
        result = RunResult(False, execution, stop_reason="exception", failure_category=category,
                           optional_failures=(str(caught),))
    return result, finalize_run(context, result), error


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        return {"unreadable": str(error)}


def run_reports(root):
    """The v5 run's own evidence: timing phases, summary and accepted best."""
    root = Path(root)
    result = {"run_root": str(root)}
    for key, relative in (("v5_timing", "reports/timing.json"), ("run_summary", "run-summary.json"),
                          ("accepted", "best/accepted.json")):
        if (root / relative).is_file():
            result[key] = _read_json(root / relative)
    result["phases"] = (result.get("v5_timing") or {}).get("phases")
    result["candidate_records"] = sorted(str(path.relative_to(root)) for path in
                                         root.glob("candidates/*/candidate.json"))
    return result


def result_record(result, reports):
    """RunResult plus the fields ``overhead_summary`` reads for workflow status."""
    value = json_value(result)
    accepted = reports.get("accepted") or {}
    candidate = accepted.get("candidate") or {}
    value.update(status=result.execution, message=result.stop_reason,
                 final_candidate_id=candidate.get("record"), final_xml=candidate.get("xml"),
                 final_report="run-summary.json" if reports.get("run_summary") else None,
                 candidates=reports.get("candidate_records", []))
    return value


def measure_primary(files, overrides, output, *, purpose, refs, tune=False, xml_path=None,
                    sidecar_path=None, cost_model_path=None, services=None, record=None):
    """Run one instrumented v5 primary workflow under ``output/project``."""
    from vericcl.execution.context import RunContext
    from vericcl.input.loader import load_request
    from vericcl.workflow_v5 import PrimaryRequest
    record = {} if record is None else record
    record.update(pipeline="v5", purpose=purpose, overrides=dict(overrides),
                  input_hashes={name: file_sha256(path) for name, path in files.items()})
    timer = PhaseTimer()
    context = None
    try:
        with instrument(timer, refs), timer.span("workflow_other"):
            inputs = load_request(files, overrides, override_input=True, purpose=purpose)
            record["input_identity"] = {"normalized_input_sha256": inputs.input_sha256,
                                        "raw_sha256": dict(record["input_hashes"])}
            snapshots = {name + ".json": path for name, path in files.items()}
            xml_text = sidecar = model = None
            if xml_path is not None:
                xml_text = Path(xml_path).read_text()
                sidecar = json.loads(Path(sidecar_path).read_text())
                snapshots.update({"schedule.xml": xml_path, "sidecar.json": sidecar_path})
            if cost_model_path is not None:
                from vericcl.performance.serialization import load_model
                model = load_model(Path(cost_model_path))
                snapshots["model.json"] = cost_model_path
            context = RunContext.create(Path(output) / "project", "overhead", "measured",
                                        input_files=snapshots,
                                        environment=dict(inputs.request_config["runtime"]["environment"]))
            result, finalized, error = execute_primary(PrimaryRequest(
                inputs, purpose=purpose, xml_text=xml_text, sidecar=sidecar, tune=tune,
                cost_model=model), context, services=services)
        reports = run_reports(context.root)
        record.update(reports)
        record["result"] = result_record(result, reports)
        record["exit_code"] = finalized.exit_code
        record["finalization_complete"] = finalized.complete
        if error is not None:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
    except (Exception, KeyboardInterrupt) as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
        if context is not None:
            record.update(run_reports(context.root))
    finally:
        record["timing"] = timer.payload()
    return record


def solve_overrides(args, sketch):
    """Controlled solver limits applied to the v5 sketch (and atom) fields."""
    overrides = parse_overrides(getattr(args, "set", None))
    controlled = {
        "sketch.solver.max_parallel_models": args.parallel,
        "sketch.solver.max_threads_per_model": args.threads,
        "sketch.solver.cpu_budget": args.parallel * args.threads,
        "sketch.solver.per_model_timeout_s": args.model_timeout,
        "sketch.budgets.max_initial_synthesis_s": args.timeout,
        # The legacy force_resolve: never adapt an indexed earlier schedule.
        "sketch.history.schedule_reuse": False,
    }
    if args.channels is not None:
        controlled["sketch.solver.max_channels"] = args.channels
    if args.bdd_backend is not None:
        controlled["sketch.solver.bdd_backend"] = args.bdd_backend
    if args.synthesis_method is not None:
        controlled["atom.synthesis_method"] = args.synthesis_method
    if not args.online:
        # CPU-only attribution: no calibration probes or traced launches.
        controlled["sketch.calibration.enabled"] = False
        controlled["sketch.trace_and_measure.enabled"] = False
    budgets = sketch.get("budgets") or {}
    if args.tune:
        rounds = (args.tuning_rounds if args.tuning_rounds is not None
                  else max(1, budgets.get("max_tuning_rounds") or 0))
        controlled["sketch.budgets.max_tuning_rounds"] = rounds
        for key, scale in (("max_repair_per_round_s", 1), ("max_repair_total_s", rounds)):
            if budgets.get(key) is None:
                controlled["sketch.budgets." + key] = args.timeout * scale
    else:
        controlled["sketch.budgets.max_tuning_rounds"] = 0
    duplicate = sorted(set(overrides) & set(controlled))
    if duplicate:
        raise ValueError("--set repeats a controlled option: " + ", ".join(duplicate))
    overrides.update(controlled)
    return overrides


def run_solve(args):
    if args.canonical_states is not None:
        import vericcl.topology.isomorphism as isomorphism
        isomorphism._MAX_CANONICAL_STATES = args.canonical_states
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    record = {"environment": environment(), "arguments": vars(args)}
    try:
        files = request_files(args, output / "request")
        overrides = solve_overrides(args, json.loads(files["sketch"].read_text()))
    except (OSError, ValueError) as error:
        record["error"] = {"type": type(error).__name__, "message": str(error)}
        record["timing"] = PhaseTimer().payload()
    else:
        measure_primary(files, overrides, output, purpose="solve", refs=registrations(),
                        tune=args.tune, cost_model_path=args.model, record=record)
    record["attribution_valid"] = args.parallel == 1
    write_json(output / "overhead.json", record)
    print(json.dumps({"output": str(output), "wall_s": record["timing"]["wall_time_s"],
                      "status": record.get("result", {}).get("status"),
                      "stop_reason": record.get("result", {}).get("stop_reason"),
                      "phases": record.get("phases"), "error": record.get("error")}), flush=True)
    return 1 if "error" in record else 0


def run_replay(args):
    from vericcl.errors import SemanticError
    from vericcl.execution.budget import StandaloneLease
    from vericcl.input.loader import load_request
    from vericcl.performance.initial import initial_cost_model
    from vericcl.performance.serialization import load_model
    from vericcl.planner.parameters import bind_artifact_partition
    from vericcl.topology import load_topology
    from vericcl.verification.symbolic_check import clear_check_cache
    from vericcl.xml import v5_lower
    from vericcl.xml.v5_model import LoweringContext
    output = Path(args.output)
    files = request_files(args, output.parent / (output.stem + "-request"))
    inputs = load_request(files, parse_overrides(args.set), override_input=True, purpose="verify")
    xml_text = Path(args.xml).read_text()
    sidecar = json.loads(Path(args.sidecar).read_text())
    try:
        inputs = bind_artifact_partition(inputs, sidecar)
    except SemanticError as error:
        raise SemanticError(str(error) + "; replay with the producing run's effective-request.json "
                            "(--effective-request) or the same --set overrides") from error
    topology = load_topology(inputs)
    model = load_model(Path(args.model)) if args.model else initial_cost_model(inputs)

    def check():
        # Correctness verdicts are cached by XML identity; each sample recomputes.
        clear_check_cache()
        lowering = LoweringContext(inputs, topology, model, StandaloneLease.after(args.timeout))
        # Module lookup at call time, so the instrumented alias is used.
        return v5_lower.check_xml_bundle(xml_text, sidecar, lowering)

    records = []
    xml_sha256 = hashlib.sha256(xml_text.encode()).hexdigest()
    payload = {"environment": environment(), "arguments": vars(args), "pipeline": "v5",
               "input_identity": input_identity(args, inputs) | {
                   "request_files": {name: str(path) for name, path in files.items()}},
               "schedule_sha256": xml_sha256, "xml_sha256": xml_sha256,
               "sidecar_sha256": file_sha256(args.sidecar), "model_version": model.version,
               "symbolic_cache": "cleared_before_each_sample", "warmups": 1,
               "records": records, "complete": False}
    write_json(args.output, payload)
    switch = TimerSwitch()
    with instrument(switch, registrations()):
        payload["warmup_outcome"] = bundle_summary(check())
        for _ in range(args.repeats):
            timer = switch.timer = PhaseTimer()
            with timer.span("replay_other"):
                bundle = check()
            records.append({"timing": timer.payload(), "outcome": bundle_summary(bundle)})
            write_json(args.output, payload, replace=True)
    payload["complete"] = True
    write_json(args.output, payload, replace=True)
    print(json.dumps({"output": args.output, "median_s": statistics.median(
        record["timing"]["wall_time_s"] for record in records),
        "can_execute": records[-1]["outcome"]["can_execute"]}), flush=True)
    return 0


def add_request_arguments(parser):
    for name in REQUEST_DOCUMENTS:
        parser.add_argument("--" + name, help="v5 %s JSON document" % name)
    parser.add_argument("--effective-request",
                        help="a v5 run's effective-request.json instead of the four documents")
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=JSON",
                        help="additional v5 input override, e.g. sketch.solver.mip_gap=0.01")


def has_request(args):
    return bool(args.effective_request) or all(getattr(args, name) for name in REQUEST_DOCUMENTS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("solve", "replay"))
    add_request_arguments(parser)
    parser.add_argument("--output", required=True)
    parser.add_argument("--xml", help="replay: admitted v5 schedule.xml")
    parser.add_argument("--sidecar", help="replay: its v5 sidecar.json")
    parser.add_argument("--model", help="cost model JSON (default: the request's initial model)")
    parser.add_argument("--bdd-backend", choices=("cudd", "autoref"),
                        help="solve: sketch.solver.bdd_backend (default: input value)")
    parser.add_argument("--repeats", type=int,
                        help="replay sample count; solve is always one invocation")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--channels", type=int, help="sketch.solver.max_channels (default: input value)")
    parser.add_argument("--timeout", type=int, default=180,
                        help="solve: max_initial_synthesis_s; replay: lease seconds per sample")
    parser.add_argument("--model-timeout", type=float, default=10)
    parser.add_argument("--canonical-states", type=int,
                        help="experimental exact-isomorphism search cap; unproved domains remain separate")
    parser.add_argument("--synthesis-method", choices=("hierarchy", "constructive_trees"),
                        help="atom.synthesis_method (default: input value)")
    parser.add_argument("--tune", action="store_true",
                        help="allow bounded tuning rounds (default: sketch rounds, at least one)")
    parser.add_argument("--tuning-rounds", type=int)
    parser.add_argument("--online", action="store_true",
                        help="solve: keep the input's calibration/trace settings (GPU jobs)")
    args = parser.parse_args(argv)
    if args.repeats is None:
        args.repeats = 1 if args.mode == "solve" else 5
    if args.mode == "solve" and args.repeats != 1:
        parser.error("solve records one invocation; use separate output directories for repeats")
    if not has_request(args):
        parser.error("the four v5 documents (--topology --sketch --atom --runtime) "
                     "or --effective-request are required")
    if args.repeats < 1 or (args.mode == "replay" and not (args.xml and args.sidecar)):
        parser.error("positive repeats and replay --xml/--sidecar are required")
    if args.mode == "solve" and (args.xml or args.sidecar):
        parser.error("solve synthesizes its own candidates; --xml/--sidecar are replay inputs")
    if args.parallel < 1 or args.threads < 1 or args.timeout <= 0 or args.model_timeout <= 0:
        parser.error("parallel, threads and timeouts must be positive")
    if args.tuning_rounds is not None and (not args.tune or args.tuning_rounds < 1):
        parser.error("--tuning-rounds requires --tune and a positive count")
    if args.canonical_states is not None and args.canonical_states < 1:
        parser.error("canonical state cap must be positive")
    return run_solve(args) if args.mode == "solve" else run_replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
