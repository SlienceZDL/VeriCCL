from fractions import Fraction

import pytest

from vericcl.errors import SemanticError


def model(alpha=0, beta=0.1, profiles=()):
    from vericcl.performance.model import CostModel, ServiceCost
    return CostModel("m0", services={"wire": (ServiceCost(alpha, beta),)},
                     resources={"wire": 10}, profiles=profiles)


def test_startup_is_paid_once_when_a_competitor_joins_and_leaves():
    from vericcl.performance.events import Executable, WorkOperation
    from vericcl.performance.simulation import predict
    schedule = Executable((WorkOperation("a", "send", "wire", 100, flow_id="a", resources=("wire",)),
                           WorkOperation("b", "send", "wire", 20, flow_id="b", resources=("wire",), not_before_us=5)))
    result = predict(schedule, model(alpha=2))
    assert result.end_times == {"a": 14, "b": 11}
    assert result.t_pred_us == 14
    assert [e.time_us for e in result.events if e.operation_id == "a" and e.kind == "startup_complete"] == [2]
    for operation in schedule.operations:
        delivered = sum(segment.rate_bytes_per_us * (segment.end_us - segment.start_us)
                        for segment in result.segments if segment.operation_id == operation.operation_id)
        assert delivered == pytest.approx(operation.payload_bytes)


def test_applicable_measured_rate_vector_replaces_equal_sharing():
    from vericcl.performance.events import Executable, WorkOperation
    from vericcl.performance.profiles import RateProfile
    from vericcl.performance.simulation import predict
    profile = RateProfile("p-ab", ("a", "b"), {"a": 8, "b": 2}, resources=("wire",), measured_at="2026-11-03T00:00:00Z")
    schedule = Executable((WorkOperation("a", "send", "wire", 100, flow_id="a", resources=("wire",)),
                           WorkOperation("b", "send", "wire", 20, flow_id="b", resources=("wire",))))
    result = predict(schedule, model(profiles=(profile,)))
    assert result.end_times["b"] == 10
    assert result.end_times["a"] == 12
    assert "profile:p-ab" in result.used_components


def test_multiple_resource_capacities_hold_for_every_progress_interval():
    from vericcl.performance.events import Executable, WorkOperation
    from vericcl.performance.model import CostModel, ServiceCost
    from vericcl.performance.simulation import predict
    cost = CostModel("shared", services={"wire": (ServiceCost(0, .1),)}, resources={"nic": 10, "pcie": 3})
    schedule = Executable((WorkOperation("a", "send", "wire", 100, flow_id="a", resources=("nic", "pcie")),
                           WorkOperation("b", "send", "wire", 100, flow_id="b", resources=("nic",))))
    result = predict(schedule, cost)
    assert result.end_times["a"] == pytest.approx(float(Fraction(100, 3)))
    assert all(s.rate_bytes_per_us <= 3 for s in result.segments if s.operation_id == "a")
    assert all(s.rate_bytes_per_us <= 10 for s in result.segments)


def test_two_endpoints_describe_one_physical_rrc_operation():
    from vericcl.performance.events import Endpoint, WorkOperation, pair_endpoints
    from vericcl.performance.model import CostModel, ServiceCost
    from vericcl.performance.simulation import predict
    operation = WorkOperation("rrc", "rrc", "rrc-cost", 100, flow_id="x", resources=("wire",))
    executable = pair_endpoints((Endpoint("send", operation), Endpoint("recv_reduce", operation)))
    result = predict(executable, CostModel("rrc-model", services={"rrc-cost": (ServiceCost(3, .2),)}, resources={"wire": 10}))
    assert result.t_pred_us == 23
    assert len(result.start_times) == 1
    with pytest.raises(SemanticError, match="endpoint"):
        pair_endpoints((Endpoint("send", operation),))


def test_dependencies_and_trailing_runtime_work_determine_full_completion():
    from vericcl.performance.events import Executable, WorkOperation
    from vericcl.performance.simulation import predict
    schedule = Executable((WorkOperation("result", "copy", "wire", 10),
                           WorkOperation("tail", "copy", "wire", 100, dependencies=("result",))),
                          output_operations=("result",))
    result = predict(schedule, model())
    assert result.output_ready_us == 1
    assert result.start_times["tail"] == 1
    assert result.t_pred_us == 11


def test_independent_local_operations_do_not_add_mixed_concurrency_penalties():
    from vericcl.performance.events import Executable, WorkOperation
    from vericcl.performance.model import CostModel, ServiceCost
    from vericcl.performance.simulation import predict
    cost = CostModel("local", services={"wire": (ServiceCost(1, .1),), "reduce": (ServiceCost(2, .2),)}, resources={"wire": 10})
    schedule = Executable((WorkOperation("network", "send", "wire", 100, flow_id="wire0", resources=("wire",)),
                           WorkOperation("local", "reduce", "reduce", 100)))
    result = predict(schedule, cost)
    assert result.end_times == {"network": 11, "local": 22}


def test_dependency_cycles_are_rejected_instead_of_returning_a_zero_time():
    from vericcl.performance.events import Executable, WorkOperation
    from vericcl.performance.simulation import predict
    schedule = Executable((WorkOperation("a", "copy", "wire", 1, dependencies=("b",)),
                           WorkOperation("b", "copy", "wire", 1, dependencies=("a",))))
    with pytest.raises(SemanticError, match="cycle"):
        predict(schedule, model())


def test_measured_startup_vector_is_selected_once_at_launch():
    from vericcl.performance.events import Executable, WorkOperation
    from vericcl.performance.profiles import RateProfile
    from vericcl.performance.simulation import predict
    profile = RateProfile("p-ab", ("a", "b"), {"a": 5, "b": 5}, resources=("wire",),
                          startup_us={"a": 2, "b": 2}, measured_at="2026-11-03T00:00:00Z")
    executable = Executable(tuple(WorkOperation(k, "send", "wire", 100, flow_id=k, resources=("wire",)) for k in ("a", "b")))
    result = predict(executable, model(alpha=1, profiles=(profile,)))
    assert result.end_times == {"a": 22, "b": 22}


def test_exact_prediction_retains_a_rational_completion_for_strict_comparison():
    from vericcl.performance.events import Executable, WorkOperation
    from vericcl.performance.simulation import predict
    executable = Executable((WorkOperation("copy", "copy", "wire", 1),))
    result = predict(executable, model(alpha=.1, beta=.2), exact=True)
    assert result.exact_completion_ratio == (3, 10)
    assert result.t_pred_us == 0.3
