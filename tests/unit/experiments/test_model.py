from pathlib import Path

import pytest

from vericcl.errors import SemanticError
from vericcl.experiments.model import load_experiment_manifest
from vericcl.input.loader import resolve_inputs
from vericcl.semantics.collective import CollectiveKind


REPO_ROOT = Path(__file__).parents[3]


def test_v100_manifest_contains_exact_matrix():
    manifest = load_experiment_manifest(
        REPO_ROOT / "exp/v100-k16-manifest.json",
        repo_root=REPO_ROOT,
    )

    assert len(manifest.cases) == 24
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
    for case in manifest.cases:
        resolved = resolve_inputs(
            case.topology_path,
            case.sketch_path,
            manifest.atom_path,
        )
        expected = (
            resolved.rank_count * resolved.hyperparameters.total_size_bytes
            if resolved.collective.kind is CollectiveKind.ALL_GATHER
            else resolved.hyperparameters.total_size_bytes
        )
        assert case.message_size_bytes == expected
        assert case.rank_count == resolved.rank_count
        assert case.slice_size_bytes == resolved.hyperparameters.slice_size_bytes
        assert resolved.solver.max_channels == 16


def test_manifest_case_order_is_deterministic():
    manifest = load_experiment_manifest(
        REPO_ROOT / "exp/v100-k16-manifest.json",
        repo_root=REPO_ROOT,
    )

    assert tuple(case.task_id for case in manifest.cases[:3]) == (
        "v100-n2g4-ag-4m",
        "v100-n2g4-ag-16m",
        "v100-n2g4-ag-64m",
    )
    assert manifest.cases[-1].task_id == "v100-n2g8-ar-2g"


def test_manifest_loader_rejects_paths_outside_repository(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"schema_version":1,"atom":"../atom.json",'
        '"topologies":["v100-n2g4"],"collectives":["ag"],'
        '"sizes":["4m"]}',
        encoding="ascii",
    )

    with pytest.raises(SemanticError, match="repository"):
        load_experiment_manifest(manifest, repo_root=tmp_path / "repo")
