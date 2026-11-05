"""Distribution integrity and example paths must survive copying the project."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


def _manifest(root, entries):
    path = root / "docs/source-provenance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "files": entries}))


def test_distribution_reports_a_modified_or_missing_file(tmp_path):
    from vericcl.artifacts.distribution import verify_source_manifest

    license_file = tmp_path / "third_party/msccl/LICENSE.txt"
    license_file.parent.mkdir(parents=True)
    license_file.write_bytes(b"original license\n")
    expected = hashlib.sha256(b"original license\n").hexdigest()
    _manifest(tmp_path, {"third_party/msccl/LICENSE.txt": expected})
    assert verify_source_manifest(tmp_path) == ()
    license_file.write_bytes(b"changed license\n")
    assert verify_source_manifest(tmp_path) == (
        {"path": "third_party/msccl/LICENSE.txt", "status": "modified"},
    )
    license_file.unlink()
    assert verify_source_manifest(tmp_path) == (
        {"path": "third_party/msccl/LICENSE.txt", "status": "missing"},
    )


@pytest.mark.parametrize("name", ["../outside", "/absolute", "a/../../outside"])
def test_distribution_rejects_paths_outside_its_root(tmp_path, name):
    from vericcl.artifacts.distribution import verify_source_manifest

    _manifest(tmp_path, {name: "0" * 64})
    with pytest.raises(ValueError, match="path"):
        verify_source_manifest(tmp_path)


def test_distribution_rejects_symlink_escape(tmp_path):
    from vericcl.artifacts.distribution import verify_source_manifest

    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"not project content")
    (root / "alias").symlink_to(outside)
    _manifest(root, {"alias": hashlib.sha256(outside.read_bytes()).hexdigest()})
    with pytest.raises(ValueError, match="path"):
        verify_source_manifest(root)


def test_example_lookup_uses_the_case_input_directory(tmp_path):
    from vericcl.resources import example_input_path

    expected = tmp_path / "examples/demo/input/topology.json"
    expected.parent.mkdir(parents=True)
    expected.write_text("{}")
    assert example_input_path("demo", "topology.json", project_root=tmp_path) == expected
    with pytest.raises(FileNotFoundError):
        example_input_path("demo", "missing.json", project_root=tmp_path)
    with pytest.raises(ValueError):
        example_input_path("../demo", "topology.json", project_root=tmp_path)
    with pytest.raises(ValueError):
        example_input_path("demo", "../output/result.json", project_root=tmp_path)


def test_shipped_runtime_and_examples_are_self_contained():
    from vericcl.artifacts.distribution import verify_source_manifest
    from vericcl.resources import example_input_path

    root = Path(__file__).resolve().parents[2]
    assert verify_source_manifest(root) == ()
    assert (root / "third_party/msccl/LICENSE.txt").is_file()
    topology = example_input_path("quickstart", "allreduce-hierarchy-outofplace/topology.json", project_root=root)
    assert isinstance(json.loads(topology.read_text()), dict)


def test_runtime_preparation_applies_patches_in_an_owned_copy(tmp_path):
    from runtime.build_runtime import prepare_runtime

    root = Path(__file__).resolve().parents[2]
    original = root / "third_party/msccl/src/init.cc"
    before = original.read_bytes()
    destination = tmp_path / "runtime-source"
    prepared = prepare_runtime(root, destination)
    metadata = json.loads((root / "runtime/msccl-trace/upstream.json").read_text())
    for relative, digest in metadata["patched_files"].items():
        assert hashlib.sha256((prepared / relative).read_bytes()).hexdigest() == digest
    assert original.read_bytes() == before
    with pytest.raises(FileExistsError):
        prepare_runtime(root, destination)


def test_runtime_prepare_cli_publishes_only_a_preparation_result(tmp_path):
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, str(root / "runtime/build_runtime.py"),
         "--output-dir", str(tmp_path / "runtime"), "--prepare-only"],
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "prepared"
    assert report["compiled"] is False
    assert Path(report["source_dir"]).is_dir()
