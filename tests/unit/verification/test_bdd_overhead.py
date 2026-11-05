from itertools import product

import pytest

import vericcl.verification.bdd_backend as module


def test_large_domain_construction_does_not_enumerate_values(monkeypatch):
    original = module.CompactBDD._value_cube
    calls = []

    def bounded(self, field, value):
        calls.append((field, value))
        assert len(calls) <= 64, "domain construction enumerated values"
        return original(self, field, value)

    monkeypatch.setattr(module.CompactBDD, "_value_cube", bounded)
    backend = module.CompactBDD({"large": 65537, "single": 1})
    relation = backend.relation(((0, 0), (65536, 0)))
    assert relation.tuples() == ((0, 0), (65536, 0))


@pytest.mark.parametrize("engine", ("dd.autoref", "dd.cudd"))
@pytest.mark.parametrize("sizes", ((1, 1), (3, 5), (8, 9), (17, 2)))
def test_domain_bounds_and_set_algebra_match_explicit_sets(monkeypatch, engine, sizes):
    pytest.importorskip(engine)
    backend = module.CompactBDD({"x": sizes[0], "y": sizes[1]}, backend=engine.split(".")[-1])
    universe = set(product(range(sizes[0]), range(sizes[1])))
    left_values = {row for row in universe if sum(row) % 2 == 0}
    right_values = {row for row in universe if row[0] % 3 == 0}
    left = backend.relation(sorted(left_values))
    right = backend.relation(sorted(right_values))
    assert set(left.complement().tuples()) == universe - left_values
    assert set(left.intersection(right).tuples()) == left_values & right_values
    assert set(left.union(right).tuples()) == left_values | right_values
    assert set(left.difference(right).tuples()) == left_values - right_values

