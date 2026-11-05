from __future__ import annotations

import importlib.abc
import json
from pathlib import Path
import sys

import pytest


pytestmark = pytest.mark.phase07


PROJECT_ROOT = Path(__file__).parents[2]
EXAMPLES = PROJECT_ROOT / "tests" / "fixtures" / "legacy-catalog"
LEGACY_PACKAGE = "ta" + "ccl"


class _RejectLegacyPackage(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == LEGACY_PACKAGE or fullname.startswith(
            LEGACY_PACKAGE + "."
        ):
            raise ImportError("legacy source package import is forbidden")
        return None


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

    from tools.generate_examples import example_documents, write_documents
    paths = write_documents(tmp_path / 'input', example_documents())
    common = ['--project-root', str(tmp_path), '--case', 'no-legacy']
    for name, path in paths.items():
        common += ['--' + name, str(path)]
    assert main(['solve', *common, '--run-id', 'solve']) == 0
    solve_root = tmp_path / 'examples/no-legacy/output/solve'
    accepted = json.loads((solve_root / 'best/accepted.json').read_text())['candidate']
    assert main(['verify', *common, '--run-id', 'verify', '--xml', str(solve_root / accepted['xml']),
                 '--sidecar', str(solve_root / accepted['sidecar'])]) == 0
    assert not any(
        module_name == LEGACY_PACKAGE
        or module_name.startswith(LEGACY_PACKAGE + ".")
        for module_name in sys.modules
    )
