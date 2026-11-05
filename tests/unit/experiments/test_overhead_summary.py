import pytest
import json


def test_timing_summary_checks_exclusive_accounting():
    from vericcl.experiments.overhead_summary import compact_timing

    with pytest.raises(ValueError, match="exclusive"):
        compact_timing({"wall_time_s": 2.0, "exclusive_s": {"phase": 1.0}})


def test_incomplete_workflow_is_never_classified_successful():
    from vericcl.experiments.overhead_summary import summarize

    data = {"arguments": {"mode": "solve"}, "timing": {
        "wall_time_s": 1.0, "exclusive_s": {"build": 1.0},
        "events": [{"error": "KeyboardInterrupt"}]}}
    assert summarize(data)["status"] == "incomplete"


def test_repeated_summary_retains_median_sample_and_failures():
    from vericcl.experiments.overhead_summary import summarize

    records = [{"timing": {"wall_time_s": value,
        "exclusive_s": {"repair": value}}, "status": "not_run"}
        for value in (3.0, 1.0, 2.0)]
    result = summarize({"arguments": {"mode": "repair"}, "records": records,
                        "complete": True})
    assert result["median_s"] == 2.0
    assert result["p95_s"] == 3.0
    assert result["median_sample_index"] == 2
    assert result["records"][2]["status"] == "not_run"


def test_repeated_online_leaf_failures_and_manifest_are_retained(tmp_path):
    from vericcl.experiments.overhead_summary import build_summary

    root = tmp_path / "worker-b" / "case"
    leaf = root / "repeat-000" / "overhead.json"
    leaf.parent.mkdir(parents=True)
    leaf.write_text(json.dumps({"arguments": {"mode": "online"},
        "result": {"failure_code": "offline_validation_rejected"}}))
    (root / "repeats.json").write_text(json.dumps({"requested": 3,
        "complete": False, "records": [{"result": "failed"}]}))
    result = build_summary(tmp_path)
    assert len(result["records"]) == 1
    record = result["records"]["worker-b/case/repeat-000/overhead.json"]
    assert record["result"]["failure_code"] == "offline_validation_rejected"
    assert "timing" not in record
    assert result["repeat_batches"]["worker-b/case/repeats.json"]["complete"] is False
