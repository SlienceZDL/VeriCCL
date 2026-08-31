import importlib

import pytest

from vericcl.errors import SolverUnavailableError
from vericcl.solver.gurobi_api import GurobiAdapter


pytestmark = pytest.mark.phase03


def test_missing_gurobi_is_reported_without_import_failure(monkeypatch):
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: None if name == "gurobipy" else object(),
    )

    assert not GurobiAdapter.available()
    with pytest.raises(SolverUnavailableError, match="gurobipy"):
        GurobiAdapter.require()


def test_gurobi_is_imported_lazily_by_require(monkeypatch):
    imported = []
    sentinel = object()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: imported.append(name) or sentinel,
    )

    assert GurobiAdapter.available()
    assert imported == []
    assert GurobiAdapter.require() is sentinel
    assert imported == ["gurobipy"]


def test_broken_gurobi_install_is_reported_as_unavailable(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())

    def fail_import(name):
        raise ImportError("broken extension")

    monkeypatch.setattr(importlib, "import_module", fail_import)

    with pytest.raises(SolverUnavailableError, match="could not be imported"):
        GurobiAdapter.require()


def test_model_creation_uses_the_supplied_explicit_environment(monkeypatch):
    calls = []

    class FakeGurobiError(Exception):
        pass

    class FakeGp:
        GurobiError = FakeGurobiError

        @staticmethod
        def Model(name, **kwargs):
            calls.append((name, kwargs))
            return object()

    monkeypatch.setattr(
        GurobiAdapter,
        "require",
        classmethod(lambda cls: FakeGp),
    )
    environment = object()

    _, explicit = GurobiAdapter.create_model(
        "explicit-model",
        environment=environment,
    )
    _, compatible = GurobiAdapter.create_model("compatible-model")

    assert explicit is not None
    assert compatible is not None
    assert calls == [
        ("explicit-model", {"env": environment}),
        ("compatible-model", {}),
    ]
