from __future__ import annotations

import importlib.abc
import json
from pathlib import Path
import sys

import pytest


pytestmark = pytest.mark.phase07


PROJECT_ROOT = Path(__file__).parents[2]
EXAMPLES = PROJECT_ROOT / "vericcl" / "examples"
LEGACY_PACKAGE = "ta" + "ccl"


class _RejectLegacyPackage(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == LEGACY_PACKAGE or fullname.startswith(
            LEGACY_PACKAGE + "."
        ):
            raise ImportError("legacy source package import is forbidden")
        return None


def _write_inputs(tmp_path):
    topology = EXAMPLES / "topo" / "two_rank.json"
    sketch_payload = json.loads(
        (EXAMPLES / "sketch" / "allreduce_8m_1m.json").read_text(
            encoding="utf-8"
        )
    )
    sketch_payload["hyperparameters"].update(
        {
            "total_size_bytes": 2048,
            "slice_size_bytes": 1024,
            "input_chunkup": 2,
            "objective_mode": "latency",
            "max_tuning_iterations": 1,
        }
    )
    sketch_payload["solver"].update(
        {
            "max_channels": 1,
            "max_parallel_models": 1,
            "max_threads_per_model": 1,
        }
    )
    atom_payload = json.loads(
        (EXAMPLES / "atom" / "default.json").read_text(encoding="utf-8")
    )
    atom_payload["strategies"]["milp"] = False
    sketch = tmp_path / "sketch.json"
    atom = tmp_path / "atom.json"
    sketch.write_text(json.dumps(sketch_payload), encoding="utf-8")
    atom.write_text(json.dumps(atom_payload), encoding="utf-8")
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


def test_public_workflow_has_no_legacy_package_dependency(tmp_path, monkeypatch):
    for module_name in tuple(sys.modules):
        if module_name == LEGACY_PACKAGE or module_name.startswith(
            LEGACY_PACKAGE + "."
        ):
            monkeypatch.delitem(sys.modules, module_name)
        if module_name == "vericcl" or module_name.startswith("vericcl."):
            monkeypatch.delitem(sys.modules, module_name)
    blocker = _RejectLegacyPackage()
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])

    from vericcl.cli.main import main

    topology, sketch, atom = _write_inputs(tmp_path)
    output_dir = tmp_path / "runs"
    solve_code = main(
        [
            "solve",
            *_common(topology, sketch, atom, output_dir, "solve"),
        ]
    )
    solve_root = output_dir / "vericcl_allreduce_2KiB_solve"
    xml_path = solve_root / "vericcl_allreduce_2KiB_final.xml"

    assert solve_code == 0
    assert xml_path.is_file()
    assert main(
        [
            "verify",
            *_common(topology, sketch, atom, output_dir, "verify"),
            "--xml",
            str(xml_path),
        ]
    ) == 0
    assert not any(
        module_name == LEGACY_PACKAGE
        or module_name.startswith(LEGACY_PACKAGE + ".")
        for module_name in sys.modules
    )
