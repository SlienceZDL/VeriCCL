import json
from pathlib import Path
import sys
from argparse import Namespace
from types import ModuleType

import pytest

from tests.unit.input.v5_support import request_documents, write_request


def test_nested_timing_partitions_wall_time_without_double_counting():
    from vericcl.experiments.overhead import PhaseTimer

    ticks = iter((0, 10, 30, 50))
    timer = PhaseTimer(clock=lambda: next(ticks))
    with timer.span("workflow", priority=0):
        with timer.span("validation", priority=10):
            pass
    assert timer.breakdown_ns() == {"workflow": 30, "validation": 20}


def test_timing_records_exceptions_without_suppressing_them():
    from vericcl.experiments.overhead import PhaseTimer

    ticks = iter((2, 9))
    timer = PhaseTimer(clock=lambda: next(ticks))
    with pytest.raises(ValueError, match="failure"):
        with timer.span("repair"):
            raise ValueError("failure")
    assert timer.breakdown_ns() == {"repair": 7}


def test_iterator_exhaustion_is_not_a_phase_error():
    from vericcl.experiments.overhead import PhaseTimer

    timer = PhaseTimer()
    with pytest.raises(StopIteration):
        with timer.span("stream"):
            next(iter(()))
    assert timer.events[0]["error"] is None


def test_instrumentation_covers_aliases_and_restores_originals():
    from vericcl.experiments.overhead import PhaseTimer, instrument

    def operation(value):
        return value + 1

    module = ModuleType("vericcl._overhead_test")
    module.first = operation
    module.second = operation
    sys.modules[module.__name__] = module
    timer = PhaseTimer()
    try:
        with instrument(timer, ((operation, "operation", 10),)):
            assert module.first(4) == 5
            assert module.second(8) == 9
        assert module.first is operation
        assert module.second is operation
        assert len(timer.events) == 2
    finally:
        del sys.modules[module.__name__]


def test_class_methods_and_lazy_streams_are_timed_per_resumption():
    from vericcl.experiments.overhead import PhaseTimer, instrument

    module = ModuleType("vericcl._overhead_stream_test")

    class Stream:
        __module__ = module.__name__

        @classmethod
        def build(cls, count):
            return cls(count)

        def __init__(self, count):
            self.count = count

        def __iter__(self):
            yield from range(self.count)

    module.Stream = Stream
    sys.modules[module.__name__] = module
    build, iterate = Stream.__dict__["build"], Stream.__dict__["__iter__"]
    timer = PhaseTimer()
    try:
        with instrument(timer, ((build.__func__, "build", 10), (iterate, "stream", 20))):
            assert list(Stream.build(3)) == [0, 1, 2]
        assert Stream.__dict__["build"] is build and Stream.__dict__["__iter__"] is iterate
        names = [event["name"] for event in timer.events]
        # One creation span, one per item, one for exhaustion and one for close.
        assert names.count("build") == 1 and names.count("stream") == 6
        assert not any(event["error"] for event in timer.events)
    finally:
        del sys.modules[module.__name__]


def test_alias_bound_while_instrumented_is_reverted():
    from vericcl.experiments.overhead import PhaseTimer, instrument

    def operation():
        return 1

    owner = ModuleType("vericcl._overhead_owner_test")
    owner.operation = operation
    sys.modules[owner.__name__] = owner
    late = ModuleType("vericcl._overhead_late_test")
    try:
        with instrument(PhaseTimer(), ((operation, "operation", 10),)):
            # A module imported during the run binds the wrapper by direct import.
            late.operation = owner.operation
            sys.modules[late.__name__] = late
            assert late.operation is not operation
        assert owner.operation is operation and late.operation is operation
    finally:
        sys.modules.pop(owner.__name__)
        sys.modules.pop(late.__name__, None)


def test_repeat_count_rejects_zero_and_warmup_is_excluded():
    from vericcl.experiments.overhead import repeated

    calls = []
    with pytest.raises(ValueError):
        repeated(lambda: None, repeats=0)
    result = repeated(lambda: calls.append(len(calls)), repeats=3)
    assert len(calls) == 4
    assert len(result["samples_s"]) == 3
    assert result["warmups"] == 1


def test_every_v5_phase_registration_resolves():
    from vericcl.experiments.overhead import PHASES, registrations

    refs = registrations(strict=True)
    labels = {label for _, label, _ in refs}
    assert {"synthesis", "model_optimize", "lowering", "symbolic_check", "prediction", "hotspots",
            "calibration", "repair_correctness", "repair_reallocation", "repair_bottleneck",
            "repair_strategy", "event_columns"} <= labels
    assert len(refs) >= len(PHASES)


def test_overhead_scripts_do_not_use_the_legacy_pipeline():
    root = Path(__file__).resolve().parents[3] / "vericcl" / "experiments"
    legacy = ("vericcl.workflow ", "vericcl.workflow\n", "import vericcl.workflow\n", "tuning.engine",
              "tuning.repair", "tuning.local_milp", "cli.solve", "cli.verify", "cli.overrides",
              "xml.emitter", "xml.buffers", "verification.pipeline", "artifacts.writer import load_",
              "calibration_xml", "cli.online")
    for name in ("overhead.py", "overhead_jobs.py", "overhead_trace.py"):
        text = (root / name).read_text()
        assert not [item for item in legacy if item in text], name


def _solve_args(paths, output, **changes):
    values = dict(topology=str(paths["topology"]), sketch=str(paths["sketch"]), atom=str(paths["atom"]),
                  runtime=str(paths["runtime"]), effective_request=None, set=[], output=str(output),
                  model=None, bdd_backend=None, parallel=1, threads=1, channels=None, timeout=30,
                  model_timeout=5.0, canonical_states=None, synthesis_method=None, tune=False,
                  tuning_rounds=None, online=False, mode="solve", repeats=1, xml=None, sidecar=None)
    values.update(changes)
    return Namespace(**values)


def test_solve_overrides_control_limits_tuning_and_online_work():
    from vericcl.experiments.overhead import solve_overrides

    paths = {name: name + ".json" for name in ("topology", "sketch", "atom", "runtime")}
    sketch = {"budgets": {"max_tuning_rounds": 3}}
    fixed = solve_overrides(_solve_args(paths, "out"), sketch)
    assert fixed["sketch.budgets.max_tuning_rounds"] == 0
    assert fixed["sketch.calibration.enabled"] is False and fixed["sketch.trace_and_measure.enabled"] is False
    assert fixed["sketch.solver.cpu_budget"] == 1 and "sketch.solver.max_channels" not in fixed
    tuned = solve_overrides(_solve_args(paths, "out", tune=True, online=True, parallel=2, threads=3,
                                        synthesis_method="hierarchy", set=["sketch.solver.mip_gap=0.01"]), sketch)
    assert tuned["sketch.budgets.max_tuning_rounds"] == 3
    assert tuned["sketch.budgets.max_repair_total_s"] == 90
    assert tuned["sketch.solver.cpu_budget"] == 6 and tuned["atom.synthesis_method"] == "hierarchy"
    assert tuned["sketch.solver.mip_gap"] == 0.01 and "sketch.calibration.enabled" not in tuned
    with pytest.raises(ValueError, match="controlled"):
        solve_overrides(_solve_args(paths, "out", set=["sketch.solver.max_threads_per_model=4"]), sketch)


@pytest.mark.scip
def test_v5_solve_and_replay_are_attributed_to_v5_phases(tmp_path, capsys):
    pytest.importorskip("pyscipopt")
    from vericcl.experiments.overhead import main
    from vericcl.experiments.overhead_summary import build_summary

    documents = request_documents("broadcast", 2)
    documents["sketch"]["solver"].update(max_channels=1, per_model_timeout_s=5)
    documents["sketch"]["budgets"].update(max_initial_synthesis_s=15, max_tuning_rounds=0)
    documents["topology"]["local_costs"] = [dict(rank=rank, copy=dict(alpha_us=.1, beta_us_per_byte=.001))
                                            for rank in range(2)]
    (tmp_path / "input").mkdir()
    paths = write_request(tmp_path / "input", documents)
    output = tmp_path / "raw" / "solve"
    argv = ["solve", "--output", str(output), "--timeout", "30", "--model-timeout", "5"]
    for name, path in paths.items():
        argv += ["--" + name, str(path)]
    assert main(argv) == 0
    record = json.loads((output / "overhead.json").read_text())
    assert "error" not in record and record["pipeline"] == "v5" and record["attribution_valid"]
    assert record["result"]["status"] == "completed" and record["exit_code"] == 0
    assert set(record["phases"]) >= {"initial_synthesis_s", "verification_s", "repair_s", "total_wall_s"}
    assert record["phases"]["tuning_rounds"] == 0
    exclusive = record["timing"]["exclusive_s"]
    assert {"synthesis", "lowering", "symbolic_check", "prediction", "artifacts"} <= set(exclusive)
    assert sum(exclusive.values()) == pytest.approx(record["timing"]["wall_time_s"])

    root = Path(record["run_root"])
    candidate = root / record["result"]["final_xml"]
    replay = tmp_path / "raw" / "replay" / "replay-best.json"
    assert main(["replay", "--effective-request", str(root / "effective-request.json"),
                 "--xml", str(candidate), "--sidecar", str(candidate.with_name("sidecar.json")),
                 "--output", str(replay), "--repeats", "2"]) == 0
    replayed = json.loads(replay.read_text())
    assert replayed["complete"] and len(replayed["records"]) == 2
    assert all(item["outcome"]["can_execute"] for item in replayed["records"])
    assert replayed["xml_sha256"] == record["accepted"]["xml_sha256"]
    assert {"verify_other", "symbolic_check"} <= set(replayed["records"][0]["timing"]["exclusive_s"])

    summary = build_summary(tmp_path / "raw")
    assert summary["records"]["solve/overhead.json"]["status"] == "completed"
    assert summary["records"]["solve/overhead.json"]["workflow"]["final_xml"] == record["result"]["final_xml"]
    assert len(summary["records"]["replay/replay-best.json"]["samples_s"]) == 2


def test_replay_requires_its_admitted_xml_and_request(tmp_path):
    from vericcl.experiments.overhead import main

    with pytest.raises(SystemExit) as error:
        main(["replay", "--output", str(tmp_path / "replay.json"), "--effective-request", "run.json"])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        main(["solve", "--output", str(tmp_path / "solve"), "--sketch", "sketch.json"])
    assert error.value.code == 2
