import json

import pytest


RANK_MAP = ({"rank": 0, "host": "worker-a", "device": 0}, {"rank": 1, "host": "worker-b", "device": 0})


def _launch(tmp_path, **evidence):
    root = tmp_path / "jobs" / "schedule-fixture"
    (root / "collected" / "worker-a" / "native").mkdir(parents=True)
    (root / "collected" / "worker-a" / "native" / "trace.rank-0.bin").write_bytes(b"raw")
    results = [{"rank": 0, "warmup_iterations": 5, "correct": True},
               {"rank": 1, "warmup_iterations": 5, "correct": True}]
    payload = {"error": None, "gpu_released": True, "cleanup_confirmed": True, "rank_results": results}
    payload.update(evidence)
    (root / "launch-result.json").write_text(json.dumps(payload))
    return root


def test_trace_replay_rejects_different_xml(tmp_path):
    from vericcl.experiments.overhead_trace import verify_xml_identity
    path = tmp_path / "schedule.xml"
    path.write_text("wrong XML")
    with pytest.raises(ValueError, match="XML identity"):
        verify_xml_identity(path, "0" * 64)


def test_raw_capture_is_rebuilt_from_retained_launch_evidence(tmp_path):
    from vericcl.experiments.overhead_trace import ReplayDriver, load_raw_capture, warmup_iterations
    raw = load_raw_capture(_launch(tmp_path), RANK_MAP)
    assert set(raw.paths) == {0} and raw.paths[0].name == "trace.rank-0.bin"
    assert raw.metadata_by_rank[1]["rank"] == 1 and raw.clock_samples == ()
    assert raw.driver_kind == "native_msccl" and raw.issues == ()
    assert warmup_iterations(raw, 9) == 5
    assert ReplayDriver(raw).capture_once(plan=None) is raw


def test_failed_or_unreleased_launch_blocks_acceptance(tmp_path):
    from vericcl.experiments.overhead_trace import load_raw_capture
    raw = load_raw_capture(_launch(tmp_path, gpu_released=False), RANK_MAP)
    assert raw.issues[0]["blocks_acceptance"] and "release" in raw.issues[0]["reason"]


def test_trace_ranks_must_agree_on_warmup(tmp_path):
    from types import SimpleNamespace
    from vericcl.experiments.overhead_trace import warmup_iterations
    raw = SimpleNamespace(metadata_by_rank={0: {"warmup_iterations": 5}, 1: {"warmup_iterations": 4}})
    with pytest.raises(ValueError, match="warmup"):
        warmup_iterations(raw, 5)
    assert warmup_iterations(SimpleNamespace(metadata_by_rank={0: {}}), 7) == 7


@pytest.mark.scip
def test_retained_capture_replays_through_the_v5_trace_pipeline(tmp_path, capsys):
    """Synthetic once-only capture laid out as a native launch directory."""
    pytest.importorskip("pyscipopt")
    import shutil
    from tests.unit.online.test_trace_causality import trace_fixture
    from vericcl.experiments.overhead_summary import build_summary
    from vericcl.experiments.overhead_trace import main
    from vericcl.verification.online.model_publication import model_json
    from vericcl.workflow_state import plain

    (tmp_path / "fixture").mkdir()
    bundle, lowering, _, raw = trace_fixture(tmp_path / "fixture")
    launch = tmp_path / "run" / "jobs" / "schedule-fixture"
    for mapping in lowering.inputs.request_config["runtime"]["rank_map"]:
        rank, base = mapping["rank"], launch / "collected" / mapping["host"] / "native"
        base.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(raw.paths[rank], base / ("trace.rank-%d.bin" % rank))
        (base / ("clock.rank-%d.log" % rank)).write_text("".join(
            "VERICCL_CLOCK_SYNC %d %d %d %d %d %d\n" % (rank, sample.gpu_ticks,
                round(sample.host_before_us * 1000), round(sample.host_after_us * 1000),
                round(sample.reference_offset_us * 1000), round(sample.reference_uncertainty_us * 1000))
            for sample in raw.clock_samples if sample.rank == rank))
    (launch / "schedule.xml").write_text(bundle.xml_text)
    (launch / "launch-result.json").write_text(json.dumps({"error": None, "gpu_released": True,
        "cleanup_confirmed": True, "rank_results": [dict(raw.metadata_by_rank[rank], rank=rank)
                                                    for rank in raw.metadata_by_rank]}))
    (tmp_path / "effective-request.json").write_text(json.dumps(plain(lowering.inputs.request_config)))
    (tmp_path / "sidecar.json").write_text(json.dumps(bundle.sidecar))
    (tmp_path / "model.json").write_text(json.dumps(model_json(lowering.model)))
    output = tmp_path / "raw" / "trace" / "overhead.json"
    assert main(["--effective-request", str(tmp_path / "effective-request.json"),
                 "--launch-directory", str(launch), "--sidecar", str(tmp_path / "sidecar.json"),
                 "--model", str(tmp_path / "model.json"), "--output", str(output), "--repeats", "2"]) == 0
    payload = json.loads(output.read_text())
    assert payload["complete"] and payload["online_acceptance"] is False
    assert payload["xml_sha256"] == bundle.xml_sha256 and payload["warmup_iterations"] == 5
    assert len(payload["records"]) == 2
    for record in payload["records"]:
        assert record["trace_status"] == "complete" and record["global_complete"]
        assert record["causality_error"] is None and record["causality_sha256"]
        assert {"trace_capture", "trace_parse", "trace_align", "trace_causality"} <= set(
            record["timing"]["exclusive_s"])
    assert build_summary(tmp_path / "raw")["records"]["trace/overhead.json"]["median_s"] > 0


def test_trace_replay_requires_launch_and_request(tmp_path):
    from vericcl.experiments.overhead_trace import main
    with pytest.raises(SystemExit) as error:
        main(["--launch-directory", str(tmp_path), "--sidecar", "sidecar.json",
              "--output", str(tmp_path / "trace.json")])
    assert error.value.code == 2
