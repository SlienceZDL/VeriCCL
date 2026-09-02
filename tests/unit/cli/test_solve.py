import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vericcl.cli.main import build_parser, main
from vericcl.cli.overrides import SemanticOverrides, resolve_semantic_overrides
from vericcl.errors import InputValidationError, SemanticError


pytestmark = pytest.mark.phase07


def _sketch(tmp_path):
    path = tmp_path / "sketch.json"
    payload = {
        "collective": {
            "operator": "allreduce",
            "root": None,
            "datatype": "float32",
            "reduction_op": "sum",
            "inplace": False,
        },
        "hyperparameters": {
            "total_size_bytes": 2048,
            "slice_size_bytes": 1024,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_solve_parser_exposes_output_and_semantic_override_options():
    args = build_parser().parse_args(
        [
            "solve",
            "--topology",
            "topology.json",
            "--sketch",
            "sketch.json",
            "--atoms",
            "atom.json",
            "--output-dir",
            "runs",
            "--run-id",
            "run-1",
            "--operator",
            "allreduce",
            "--total-size-bytes",
            "4096",
            "--slice-size-bytes",
            "1024",
            "--inplace",
            "--override-input",
            "--timeout-s",
            "12",
        ]
    )

    assert args.output_dir == Path("runs")
    assert args.run_id == "run-1"
    assert args.operator == "allreduce"
    assert args.total_size_bytes == 4096
    assert args.slice_size_bytes == 1024
    assert args.inplace is True
    assert args.override_input is True
    assert args.timeout_s == 12.0


def test_conflicting_semantic_value_requires_explicit_override(tmp_path):
    sketch = _sketch(tmp_path)
    original = sketch.read_bytes()

    with pytest.raises(InputValidationError, match="conflicts"):
        resolve_semantic_overrides(
            sketch,
            SemanticOverrides(total_size_bytes=4096),
            allow_override=False,
            output_dir=tmp_path,
        )

    assert sketch.read_bytes() == original


def test_explicit_override_writes_effective_sketch_without_mutating_input(
    tmp_path,
):
    sketch = _sketch(tmp_path)
    original = sketch.read_bytes()

    effective = resolve_semantic_overrides(
        sketch,
        SemanticOverrides(
            total_size_bytes=4096,
            slice_size_bytes=2048,
            inplace=True,
        ),
        allow_override=True,
        output_dir=tmp_path / "effective",
    )

    payload = json.loads(effective.read_text(encoding="utf-8"))
    assert payload["hyperparameters"]["total_size_bytes"] == 4096
    assert payload["hyperparameters"]["slice_size_bytes"] == 2048
    assert payload["collective"]["inplace"] is True
    assert sketch.read_bytes() == original


@pytest.mark.parametrize(
    "values",
    (
        {"operator": "invalid"},
        {"total_size_bytes": 0},
        {"slice_size_bytes": True},
        {"root": -1},
        {"inplace": "yes"},
    ),
)
def test_semantic_override_model_rejects_invalid_values(values):
    with pytest.raises(SemanticError, match="semantic override"):
        SemanticOverrides(**values)


@pytest.mark.parametrize(
    "payload",
    (
        [],
        {},
        {"collective": {}, "hyperparameters": []},
    ),
)
def test_override_rejects_invalid_sketch_structure(tmp_path, payload):
    sketch = tmp_path / "invalid.json"
    sketch.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InputValidationError, match="sketch"):
        resolve_semantic_overrides(
            sketch,
            SemanticOverrides(total_size_bytes=4096),
            allow_override=True,
            output_dir=tmp_path / "output",
        )


def test_matching_semantic_override_reuses_original_sketch(tmp_path):
    sketch = _sketch(tmp_path)

    effective = resolve_semantic_overrides(
        sketch,
        SemanticOverrides(total_size_bytes=2048),
        allow_override=False,
        output_dir=tmp_path / "unused",
    )

    assert effective == sketch
    assert not (tmp_path / "unused").exists()


def test_override_refuses_to_replace_existing_effective_sketch(tmp_path):
    sketch = _sketch(tmp_path)
    output = tmp_path / "effective"
    output.mkdir()
    destination = output / "effective-sketch.json"
    destination.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        resolve_semantic_overrides(
            sketch,
            SemanticOverrides(total_size_bytes=4096),
            allow_override=True,
            output_dir=output,
        )

    assert destination.read_text(encoding="utf-8") == "preserve"


def test_missing_input_file_returns_fatal_input_exit(tmp_path, capsys):
    code = main(
        [
            "solve",
            "--topology",
            str(tmp_path / "missing-topology.json"),
            "--sketch",
            str(tmp_path / "missing-sketch.json"),
            "--atoms",
            str(tmp_path / "missing-atom.json"),
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "missing" in captured.err.lower()


def test_clean_solve_prints_one_summary_and_returns_success(monkeypatch, capsys):
    result = SimpleNamespace(
        status="feasible",
        message="complete",
        final_candidate_id="candidate-0",
        final_xml=Path("/tmp/final.xml"),
        layout=SimpleNamespace(root=Path("/tmp/run")),
        candidates=(),
    )
    monkeypatch.setattr("vericcl.cli.solve.execute_solve", lambda context: result)
    monkeypatch.setattr("vericcl.cli.solve.require_input_files", lambda *args: None)

    code = main(
        [
            "solve",
            "--topology",
            "topology.json",
            "--sketch",
            "sketch.json",
            "--atoms",
            "atom.json",
            "--run-id",
            "test",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    assert "status=feasible" in captured.out


def test_v100_experiment_document_uses_safe_fixed_contract():
    text = Path("docs/experiments/v100-k16.md").read_text(encoding="utf-8")

    assert "NCCL_IB_DISABLE=1" in text
    assert "--config" in text
    assert "--resume" in text
    assert (
        "/home/zdl/VeriCCL-experiments/2026-09-01-v100-k16" in text
    )
    assert "/home/cc" not in text
