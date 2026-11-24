import json
from pathlib import Path

import pytest

from vericcl.errors import SemanticError
from vericcl.experiments.model import case_request_files, load_experiment_manifest
from vericcl.input.loader import load_request
from vericcl.semantics.collective import CollectiveKind


REPO_ROOT = Path(__file__).parents[3]
FIXTURES = REPO_ROOT / "tests/fixtures/experiments-v5"


def test_v100_manifest_contains_exact_matrix():
    manifest = load_experiment_manifest(
        FIXTURES / "v100-k16-manifest.json",
        repo_root=REPO_ROOT,
    )

    assert len(manifest.cases) == 24
    assert manifest.atom_path == FIXTURES / "atom.json"
    assert {case.topology_name for case in manifest.cases} == {
        "v100-n2g4",
        "v100-n2g8",
    }
    assert {case.collective_label for case in manifest.cases} == {"ag", "ar"}
    assert {case.size_label for case in manifest.cases} == {
        "4m",
        "16m",
        "64m",
        "256m",
        "1g",
        "2g",
    }
    assert len({case.task_id for case in manifest.cases}) == 24
    sizes = {"4m": 4 << 20, "16m": 16 << 20, "64m": 64 << 20, "256m": 256 << 20, "1g": 1 << 30, "2g": 2 << 30}
    for case in manifest.cases:
        resolved = load_request(case_request_files(case, manifest.atom_path))
        expected = (
            resolved.rank_count * resolved.hyperparameters.total_size_bytes
            if resolved.collective.kind is CollectiveKind.ALL_GATHER
            else resolved.hyperparameters.total_size_bytes
        )
        assert case.message_size_bytes == expected == sizes[case.size_label]
        assert case.rank_count == resolved.rank_count == {"v100-n2g4": 8, "v100-n2g8": 16}[case.topology_name]
        assert case.slice_size_bytes == resolved.hyperparameters.slice_size_bytes
        assert case.nccl_buffsize_bytes == 2 * case.slice_size_bytes
        assert resolved.solver.max_channels == 16
        assert resolved.collective.inplace is True
        assert resolved.collective.datatype == "float32"
        assert resolved.selected_synthesis_method == "constructive_trees"


def test_manifest_case_order_is_deterministic():
    manifest = load_experiment_manifest(
        FIXTURES / "v100-k16-manifest.json",
        repo_root=REPO_ROOT,
    )

    assert tuple(case.task_id for case in manifest.cases[:3]) == (
        "v100-n2g4-ag-4m",
        "v100-n2g4-ag-16m",
        "v100-n2g4-ag-64m",
    )
    assert manifest.cases[-1].task_id == "v100-n2g8-ar-2g"


def test_a100_manifest_resolves():
    manifest = load_experiment_manifest(
        FIXTURES / "a100-k16-manifest.json",
        repo_root=REPO_ROOT,
    )

    assert len(manifest.cases) == 12
    assert {case.rank_count for case in manifest.cases} == {16}


def test_manifest_loader_rejects_paths_outside_repository(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"schema_version":2,"atom":"../atom.json","cases":[]}',
        encoding="ascii",
    )

    with pytest.raises(SemanticError, match="repository"):
        load_experiment_manifest(manifest, repo_root=tmp_path / "repo")


def test_manifest_loader_rejects_legacy_schema(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "atom": "atom.json", "topologies": ["v100-n2g4"],
                    "collectives": ["ag"], "sizes": ["4m"]}),
        encoding="ascii",
    )

    with pytest.raises(SemanticError, match="fields are invalid"):
        load_experiment_manifest(manifest, repo_root=tmp_path)
    manifest.write_text(json.dumps({"schema_version": 1, "atom": "atom.json", "cases": []}), encoding="ascii")
    with pytest.raises(SemanticError, match="schema is unsupported"):
        load_experiment_manifest(manifest, repo_root=tmp_path)


def _write_manifest(tmp_path, **overrides):
    case = json.loads((FIXTURES / "v100-k16-manifest.json").read_text())["cases"][0]
    case.update(overrides)
    for key in ("topology", "sketch", "runtime"):
        source = REPO_ROOT / case[key]
        target = tmp_path / case[key]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    atom = tmp_path / "atom.json"
    atom.write_bytes((FIXTURES / "atom.json").read_bytes())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 2, "atom": "atom.json", "cases": [case]}))
    return manifest


def test_manifest_collective_must_match_sketch(tmp_path):
    manifest = _write_manifest(tmp_path, collective_label="ar")

    with pytest.raises(SemanticError, match="collective differs"):
        load_experiment_manifest(manifest, repo_root=tmp_path)


def test_manifest_rejects_invalid_v5_case(tmp_path):
    manifest = _write_manifest(tmp_path)
    runtime_path = tmp_path / json.loads(manifest.read_text())["cases"][0]["runtime"]
    runtime = json.loads(runtime_path.read_text())
    del runtime["environment"]["NCCL_BUFFSIZE"]
    runtime_path.write_text(json.dumps(runtime))

    with pytest.raises(SemanticError, match="not a valid v5 request"):
        load_experiment_manifest(manifest, repo_root=tmp_path)
