import hashlib
import json
from pathlib import Path

import pytest

import vericcl.workflow as workflow_module
from vericcl.workflow import RunContext, execute_solve, execute_verify


pytestmark = pytest.mark.phase07


PROJECT_ROOT = Path(__file__).parents[2]
EXAMPLES = PROJECT_ROOT / "vericcl" / "examples"


def _write_constructive_inputs(tmp_path):
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
    sketch_payload["solver"]["max_channels"] = 1
    sketch_payload["solver"]["max_parallel_models"] = 1
    sketch_payload["solver"]["max_threads_per_model"] = 1
    sketch = tmp_path / "sketch.json"
    sketch.write_text(json.dumps(sketch_payload), encoding="utf-8")
    return topology, sketch, atom


def test_solve_workflow_writes_complete_lineage_and_final_alias(tmp_path):
    topology, sketch, atom = _write_constructive_inputs(tmp_path)
    result = execute_solve(
        RunContext(
            topology_path=topology,
            sketch_path=sketch,
            atom_path=atom,
            output_base=tmp_path / "runs",
            run_id="solve-0001",
            solver_version="test-solver",
            model_version="test-model",
            environment_signature="test-environment",
        )
    )

    assert result.final_candidate_id is not None
    assert result.final_xml is not None and result.final_xml.is_file()
    assert result.final_report is not None and result.final_report.is_file()
    assert result.layout.resolved_input.is_file()
    assert result.layout.summary.is_file()
    summary = json.loads(result.layout.summary.read_text(encoding="utf-8"))
    assert summary["final_selection"]["candidate_id"] == (
        result.final_candidate_id
    )
    assert summary["final_selection"]["xml_path"] == result.final_xml.name
    assert len(summary["candidates"]) == len(result.candidates)
    assert all("optimal" not in item["xml_path"] for item in summary["candidates"])
    for item in summary["candidates"]:
        if item["xml_path"] is None:
            continue
        xml_path = result.layout.root / item["xml_path"]
        report_path = result.layout.root / item["report_path"]
        assert hashlib.sha256(xml_path.read_bytes()).hexdigest() == item[
            "xml_sha256"
        ]
        assert hashlib.sha256(report_path.read_bytes()).hexdigest() == item[
            "report_sha256"
        ]


def test_verify_workflow_reconstructs_schedule_and_uses_same_validation(tmp_path):
    topology, sketch, atom = _write_constructive_inputs(tmp_path)
    solved = execute_solve(
        RunContext(
            topology_path=topology,
            sketch_path=sketch,
            atom_path=atom,
            output_base=tmp_path / "runs",
            run_id="source",
        )
    )
    selected = next(
        item
        for item in solved.candidates
        if item.candidate_id == solved.final_candidate_id
    )

    verified = execute_verify(
        RunContext(
            topology_path=topology,
            sketch_path=sketch,
            atom_path=atom,
            output_base=tmp_path / "runs",
            run_id="verify",
            xml_path=selected.xml_path,
            sidecar_path=selected.schedule_path,
        )
    )

    assert verified.final_candidate_id == selected.candidate_id
    assert verified.final_xml is not None
    summary = json.loads(verified.layout.summary.read_text(encoding="utf-8"))
    assert summary["mode"] == "verify"
    assert summary["candidates"][0]["validation"]["semantic"] == "valid"
    assert summary["candidates"][0]["validation"]["xml"] == "valid"


def test_workflow_refuses_to_overwrite_a_previous_run(tmp_path):
    topology, sketch, atom = _write_constructive_inputs(tmp_path)
    context = RunContext(
        topology_path=topology,
        sketch_path=sketch,
        atom_path=atom,
        output_base=tmp_path / "runs",
        run_id="same",
    )
    execute_solve(context)

    with pytest.raises(FileExistsError, match="non-empty"):
        execute_solve(context)


def test_invalid_candidate_writes_report_and_sidecar_but_no_xml(
    tmp_path,
    monkeypatch,
):
    topology, sketch, atom = _write_constructive_inputs(tmp_path)
    original = workflow_module.validate_and_lower_candidate

    def force_invalid(schedule, inputs, topology_value):
        return original(
            schedule.__class__(
                schedule.schedule_id,
                schedule.transfers,
                schedule.final_state_ids,
                schedule.rank_count + 1,
                schedule.slice_count,
                schedule.slice_size_bytes,
                schedule.metadata,
            ),
            inputs,
            topology_value,
        )

    monkeypatch.setattr(
        workflow_module,
        "validate_and_lower_candidate",
        force_invalid,
    )
    result = execute_solve(
        RunContext(
            topology_path=topology,
            sketch_path=sketch,
            atom_path=atom,
            output_base=tmp_path / "runs",
            run_id="invalid",
        )
    )

    assert result.final_candidate_id is None
    assert result.final_xml is None
    assert result.candidates
    assert all(item.xml_path is None for item in result.candidates)
    assert all(item.report_path.is_file() for item in result.candidates)
    assert all(item.schedule_path.is_file() for item in result.candidates)
    summary = json.loads(result.layout.summary.read_text(encoding="utf-8"))
    assert summary["final_selection"] is None
    assert all(item["xml_path"] is None for item in summary["candidates"])


def test_workflow_wall_clock_budget_includes_input_resolution(
    tmp_path,
    monkeypatch,
):
    topology, sketch, atom = _write_constructive_inputs(tmp_path)
    timestamps = iter((0.0, 2.0))
    monkeypatch.setattr(workflow_module, "_monotonic", lambda: next(timestamps))

    with pytest.raises(TimeoutError, match="input resolution"):
        execute_solve(
            RunContext(
                topology_path=topology,
                sketch_path=sketch,
                atom_path=atom,
                output_base=tmp_path / "runs",
                run_id="timeout",
                timeout_s=1.0,
            )
        )
