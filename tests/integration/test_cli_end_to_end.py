import json
from pathlib import Path

import pytest

from vericcl.cli.main import main


pytestmark = pytest.mark.phase07


PROJECT_ROOT = Path(__file__).parents[2]
EXAMPLES = PROJECT_ROOT / "vericcl" / "examples"


def _inputs(tmp_path):
    topology = EXAMPLES / "topo" / "two_rank.json"
    atom_payload = json.loads(
        (EXAMPLES / "atom" / "default.json").read_text(encoding="utf-8")
    )
    atom_payload["strategies"]["milp"] = False
    atom = tmp_path / "atom.json"
    atom.write_text(json.dumps(atom_payload), encoding="utf-8")
    sketch_payload = json.loads(
        (EXAMPLES / "sketch" / "allreduce_8m_1m.json").read_text(
            encoding="utf-8"
        )
    )
    sketch_payload["hyperparameters"]["total_size_bytes"] = 2 * 1024 * 1024
    sketch_payload["hyperparameters"]["input_chunkup"] = 2
    sketch_payload["hyperparameters"]["objective_mode"] = "latency"
    sketch_payload["hyperparameters"]["max_tuning_iterations"] = 1
    sketch_payload["solver"]["max_channels"] = 1
    sketch_payload["solver"]["max_parallel_models"] = 1
    sketch_payload["solver"]["max_threads_per_model"] = 1
    sketch = tmp_path / "sketch.json"
    sketch.write_text(json.dumps(sketch_payload), encoding="utf-8")
    return topology, sketch, atom


def _common(topology, sketch, atom, output_dir, run_id):
    return [
        "--topology",
        str(topology),
        "--sketch",
        str(sketch),
        "--atoms",
        str(atom),
        "--output-dir",
        str(output_dir),
        "--run-id",
        run_id,
    ]


def test_public_cli_solves_then_verifies_final_xml(tmp_path, capsys):
    topology, sketch, atom = _inputs(tmp_path)
    output_dir = tmp_path / "runs"

    solve_code = main(
        ["solve", *_common(topology, sketch, atom, output_dir, "solve")]
    )
    solve_output = capsys.readouterr()
    solve_root = output_dir / "vericcl_allreduce_2MiB_solve"
    final_xml = solve_root / "vericcl_allreduce_2MiB_final.xml"
    final_sidecar = solve_root / "vericcl_allreduce_2MiB_final.schedule.json"

    assert solve_code == 0
    assert solve_output.err == ""
    assert final_xml.is_file()
    assert final_sidecar.is_file()
    solve_summary = json.loads((solve_root / "run-summary.json").read_text())
    assert solve_summary["planning_mode"] == "direct"
    assert solve_summary["search_model_count_total"] >= 0
    assert solve_summary["verification_time_s"] > 0.0
    assert solve_summary["total_wall_clock_time_s"] == solve_summary["elapsed_s"]

    verify_code = main(
        [
            "verify",
            *_common(topology, sketch, atom, output_dir, "verify"),
            "--xml",
            str(final_xml),
        ]
    )
    verify_output = capsys.readouterr()

    assert verify_code == 0
    assert verify_output.err == ""
    summary = json.loads(
        (
            output_dir
            / "vericcl_allreduce_2MiB_verify"
            / "run-summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["mode"] == "verify"
    assert summary["final_selection"] is not None
    assert summary["route_model_count"] == 0
    assert summary["search_model_count_total"] == 0
    assert summary["verification_time_s"] > 0.0

    tune_code = main(
        [
            "verify",
            *_common(topology, sketch, atom, output_dir, "verify-tune"),
            "--xml",
            str(final_xml),
            "--tune",
        ]
    )
    tune_output = capsys.readouterr()

    assert tune_code == 0
    assert tune_output.err == ""
    tuned_summary = json.loads(
        (
            output_dir
            / "vericcl_allreduce_2MiB_verify-tune"
            / "run-summary.json"
        ).read_text(encoding="utf-8")
    )
    assert "tuning=" in tuned_summary["message"]
    assert tuned_summary["final_selection"] is not None
    tuned_reports = tuple(
        json.loads(
            (
                output_dir
                / "vericcl_allreduce_2MiB_verify-tune"
                / item["report_path"]
            ).read_text(encoding="utf-8")
        )
        for item in tuned_summary["candidates"]
    )
    assert all(report["verification_time_s"] > 0.0 for report in tuned_reports)
    assert tuned_summary["verification_time_s"] >= sum(
        report["verification_time_s"] for report in tuned_reports
    )
    assert all(
        isinstance(report["tuning_strategy"]["kind"], str)
        for report in tuned_reports
    )


def test_cli_explicit_override_changes_resolved_snapshot_only(tmp_path, capsys):
    topology, sketch, atom = _inputs(tmp_path)
    output_dir = tmp_path / "runs"
    original = sketch.read_bytes()

    code = main(
        [
            "solve",
            *_common(topology, sketch, atom, output_dir, "override"),
            "--total-size-bytes",
            str(4 * 1024 * 1024),
            "--override-input",
        ]
    )
    captured = capsys.readouterr()

    assert code == 0
    assert captured.err == ""
    resolved = json.loads(
        (
            output_dir
            / "vericcl_allreduce_4MiB_override"
            / "resolved-input.json"
        ).read_text(encoding="utf-8")
    )
    assert resolved["sketch"]["hyperparameters"]["total_size_bytes"] == (
        4 * 1024 * 1024
    )
    assert sketch.read_bytes() == original
