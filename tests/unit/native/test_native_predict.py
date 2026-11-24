"""The native exact replay must equal the Python reference field for field."""
from fractions import Fraction

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from vericcl import native
from vericcl.errors import SemanticError
from vericcl.performance.events import Executable, WorkOperation
from vericcl.performance.model import CostModel, ServiceCost
from vericcl.performance.profiles import RateProfile
from vericcl.performance.simulation import predict

pytestmark = pytest.mark.skipif(native.library() is None and native.mode() != "require",
                                reason="native core not built (python tools/build_native.py)")

DECIMALS = (0.0, 0.1, 0.25, 1.0, 1.5, 3.0, 7.3, 1e-05, 0.003, 12.75)
BETAS = (0.001, 0.0025, 0.01, 0.1, 1e-05, 0.3333)
CAPACITIES = (0.5, 1.0, 50.0, 100.0, 1000.0, 333.3)


def reference(executable, model, monkeypatch):
    monkeypatch.setenv("VERICCL_CORE", "off")
    try:
        return "ok", predict(executable, model, exact=True)
    except SemanticError as error:
        return "error", str(error)
    finally:
        monkeypatch.setenv("VERICCL_CORE", "require")


def accelerated(executable, model, monkeypatch):
    monkeypatch.setenv("VERICCL_CORE", "require")
    try:
        return "ok", predict(executable, model, exact=True)
    except SemanticError as error:
        return "error", str(error)


def assert_same(executable, model, monkeypatch):
    expected = reference(executable, model, monkeypatch)
    actual = accelerated(executable, model, monkeypatch)
    assert actual[0] == expected[0], (actual, expected)
    if expected[0] == "error":
        assert actual[1] == expected[1]
        return
    left, right = actual[1], expected[1]
    assert left == right
    # Mapping insertion order is part of what downstream serializers see.
    assert list(left.start_times) == list(right.start_times)
    assert list(left.end_times) == list(right.end_times)


@st.composite
def cases(draw):
    count = draw(st.integers(1, 18))
    names = draw(st.permutations(["op-%02d" % index for index in range(count)]))
    operations = []
    flows = ["f%d" % index for index in range(draw(st.integers(1, count)))]
    for position in range(count):
        kind = draw(st.sampled_from(("send", "rrc", "copy", "reduce", "nop")))
        dependencies = tuple(sorted(set(draw(st.lists(st.sampled_from(names[:position]), max_size=3)))
                                    if position else ()))
        not_before = draw(st.sampled_from(DECIMALS))
        if kind == "nop":
            operations.append(WorkOperation(names[position], "nop", None, 0, dependencies, not_before_us=not_before))
            continue
        resources = ()
        flow = None
        if kind in {"send", "rrc"}:
            resources = tuple(sorted(set(draw(st.lists(st.sampled_from(("r0", "r1", "r2")), max_size=2)))))
            flow = draw(st.sampled_from(flows))
        operations.append(WorkOperation(names[position], kind, draw(st.sampled_from(("k0", "k1", "k2"))),
                                        draw(st.integers(1, 4096)), dependencies, resources, flow, not_before,
                                        draw(st.sampled_from((None, None, "estimated")))))
    services = {}
    for key in ("k0", "k1", "k2"):
        if draw(st.integers(0, 19)) == 0:
            continue  # missing service component
        upper = draw(st.sampled_from((None,) * 5 + (2048,)))
        services[key] = (ServiceCost(draw(st.sampled_from(DECIMALS)), draw(st.sampled_from(BETAS)),
                                     draw(st.sampled_from(("initial", "estimated"))), 1, upper),)
    resources = {name: draw(st.sampled_from(CAPACITIES)) for name in ("r0", "r1", "r2")
                 if draw(st.integers(0, 15))}
    profiles = []
    communication = [op for op in operations if op.kind in {"send", "rrc"}]
    for number in range(draw(st.integers(0, 2))):
        if not communication:
            break
        members = draw(st.lists(st.sampled_from(communication), min_size=1, max_size=3, unique_by=lambda op: op.operation_id))
        mode = sorted({op.flow_id for op in members})
        if len(mode) != len(members):
            continue
        profile_resources = sorted({resource for op in members for resource in op.resources})
        rates = {flow: draw(st.sampled_from(CAPACITIES)) for flow in mode}
        startup = {flow: draw(st.sampled_from(DECIMALS)) for flow in mode if draw(st.booleans())}
        membership = ({op.flow_id: op.resources for op in members}
                      if draw(st.booleans()) and all(op.resources for op in members) else {})
        profiles.append(RateProfile("p%d" % number, tuple(mode), rates, tuple(profile_resources), source="estimated",
                                    startup_us=startup, flow_resources=membership))
    outputs = tuple(draw(st.lists(st.sampled_from(names), min_size=1, max_size=3, unique=True)))
    return Executable(tuple(operations), outputs), CostModel("m", services, resources, tuple(profiles))


@settings(max_examples=800, deadline=None, suppress_health_check=[HealthCheck.too_slow,
                                                                   HealthCheck.function_scoped_fixture])
@given(case=cases())
def test_native_replay_matches_python_reference(case, monkeypatch):
    assert_same(*case, monkeypatch)


def test_native_rounding_matches_python_integer_division(monkeypatch):
    # Thirds and sevenths make long binary expansions: the event times are the
    # correctly rounded floats of exact rationals.
    operations = (WorkOperation("a", "copy", "k", 7), WorkOperation("b", "copy", "k", 11, ("a",)),
                  WorkOperation("c", "send", "k", 13, ("b",), ("r",), "f"))
    model = CostModel("m", {"k": (ServiceCost(0.1, 0.3333),)}, {"r": 3.0})
    assert_same(Executable(operations), model, monkeypatch)
    monkeypatch.setenv("VERICCL_CORE", "require")
    result = predict(Executable(operations), model, exact=True)
    assert Fraction(*result.exact_completion_ratio) == Fraction(result.exact_completion_ratio[0],
                                                                 result.exact_completion_ratio[1])
    exact = Fraction(*result.exact_completion_ratio)
    assert result.t_pred_us == exact.numerator / exact.denominator


def test_native_reports_the_reference_error_for_shared_flows(monkeypatch):
    operations = (WorkOperation("a", "send", "k", 8, (), ("r",), "same"),
                  WorkOperation("b", "send", "k", 8, (), ("r",), "same"))
    model = CostModel("m", {"k": (ServiceCost(0, 0.01),)}, {"r": 10.0})
    assert accelerated(Executable(operations), model, monkeypatch) == (
        "error", "concurrent operations share a flow identity; serialize their TB work")


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow,
                                                                   HealthCheck.function_scoped_fixture])
@given(case=cases())
def test_summary_mode_keeps_totals_and_matches_reference(case, monkeypatch):
    executable, model = case
    results = {}
    for selected in ("off", "require"):
        monkeypatch.setenv("VERICCL_CORE", selected)
        try:
            full = predict(executable, model, exact=True)
            summary = predict(executable, model, exact=True, detail=False)
        except SemanticError as error:
            results[selected] = str(error)
            continue
        assert summary.events == () and summary.segments == () and not summary.start_times
        for field in ("t_pred_us", "output_ready_us", "exact_completion_ratio", "used_components",
                      "missing_components", "unavailable_operations", "model_coverage", "unmeasured_modes"):
            assert getattr(summary, field) == getattr(full, field), field
        results[selected] = summary
    assert results["off"] == results["require"]
