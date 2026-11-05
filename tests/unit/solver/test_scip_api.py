import importlib

import pytest

from vericcl.errors import SolverUnavailableError
from vericcl.solver.scip_api import ScipAdapter


pytestmark = pytest.mark.phase03


def test_missing_scip_is_reported_without_import_failure(monkeypatch):
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: None if name == "pyscipopt" else object(),
    )

    assert not ScipAdapter.available()
    with pytest.raises(SolverUnavailableError, match="SCIP"):
        ScipAdapter.require()


def test_scip_is_imported_lazily_by_require(monkeypatch):
    imported = []
    sentinel = object()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: imported.append(name) or sentinel,
    )

    assert ScipAdapter.available()
    assert imported == []
    assert ScipAdapter.require() is sentinel
    assert imported == ["pyscipopt"]


def test_broken_scip_install_is_reported_as_unavailable(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())

    def fail_import(name):
        raise ImportError("broken extension")

    monkeypatch.setattr(importlib, "import_module", fail_import)

    with pytest.raises(SolverUnavailableError, match="could not be loaded"):
        ScipAdapter.require()


def test_each_model_owns_a_distinct_native_engine(monkeypatch):
    calls = []
    class FakeEngine:
        def __init__(self, name):
            calls.append(name)
        def hideOutput(self):
            pass
    class FakeModule:
        Model = FakeEngine
    monkeypatch.setattr(ScipAdapter, 'require', lambda: FakeModule)
    _, first = ScipAdapter.create_model('first')
    _, second = ScipAdapter.create_model('second')
    assert first.engine is not second.engine
    assert calls == ['first', 'second']
