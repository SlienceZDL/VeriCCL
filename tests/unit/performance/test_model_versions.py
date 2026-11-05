import pytest

from vericcl.errors import SemanticError


def test_component_update_preserves_the_original_and_changes_prediction_identity():
    from vericcl.performance.events import Executable, WorkOperation
    from vericcl.performance.model import CostModel, ServiceCost
    from vericcl.performance.simulation import predict
    before = CostModel("initial", services={"copy": (ServiceCost(1, .1),)})
    measured = ServiceCost(2, .2, source="measured", measured_at="2026-11-03T00:00:00Z")
    after, changes = before.with_updates("measured", services={"copy": (measured,)})
    executable = Executable((WorkOperation("copy", "copy", "copy", 10),))
    assert predict(executable, before).t_pred_us == 2
    assert predict(executable, after).t_pred_us == 4
    assert predict(executable, after).model_version == "measured"
    assert changes[0]["component"] == "service:copy"
    assert before.services["copy"][0].source == "initial"


def test_unknown_component_keeps_unrelated_predictions_and_blocks_full_tpred():
    from vericcl.performance.events import Executable, WorkOperation
    from vericcl.performance.model import CostModel, ServiceCost
    from vericcl.performance.simulation import predict
    cost = CostModel("partial", services={"known": (ServiceCost(1, .1),)})
    executable = Executable((WorkOperation("known", "copy", "known", 10),
                             WorkOperation("unknown", "copy", "missing", 10),
                             WorkOperation("dependent", "copy", "known", 10, dependencies=("unknown",))))
    result = predict(executable, cost)
    assert result.t_pred_us is None
    assert result.end_times == {"known": 2}
    assert set(result.unavailable_operations) == {"unknown", "dependent"}
    assert "service:missing" in result.missing_components


def test_unknown_competitor_also_limits_shared_resource_conclusions():
    from vericcl.performance.events import Executable, WorkOperation
    from vericcl.performance.model import CostModel, ServiceCost
    from vericcl.performance.simulation import predict
    cost = CostModel("partial", services={"known": (ServiceCost(0, .1),)}, resources={"nic": 10})
    executable = Executable((WorkOperation("a", "send", "known", 10, flow_id="a", resources=("nic",)),
                             WorkOperation("b", "send", "missing", 10, flow_id="b", resources=("nic",)),
                             WorkOperation("c", "copy", "known", 10)))
    result = predict(executable, cost)
    assert set(result.unavailable_operations) == {"a", "b"}
    assert result.end_times == {"c": 1}


def test_size_regime_is_selected_from_the_full_payload_not_remaining_bytes():
    from vericcl.performance.events import Executable, WorkOperation
    from vericcl.performance.model import CostModel, ServiceCost
    from vericcl.performance.simulation import predict
    cost = CostModel("pieces", services={"copy": (ServiceCost(10, 1, min_bytes=1, max_bytes=10),
                                                 ServiceCost(1, .1, min_bytes=11, max_bytes=100))})
    result = predict(Executable((WorkOperation("large", "copy", "copy", 100),
                                 WorkOperation("tick", "copy", "copy", 1, not_before_us=5))), cost)
    assert result.end_times["large"] == 11
    assert result.end_times["tick"] == 16


@pytest.mark.parametrize("alpha,beta", [(-1, 1), (0, 0), (0, -1), (float("nan"), 1)])
def test_costs_reject_zero_slope_and_nonfinite_or_negative_parameters(alpha, beta):
    from vericcl.performance.model import ServiceCost
    with pytest.raises(SemanticError):
        ServiceCost(alpha, beta)


def test_measured_profile_cannot_silently_exceed_declared_resource_capacity():
    from vericcl.performance.events import Executable, WorkOperation
    from vericcl.performance.model import CostModel, ServiceCost
    from vericcl.performance.profiles import RateProfile
    from vericcl.performance.simulation import predict
    cost = CostModel("conflict", services={"wire": (ServiceCost(0, .1),)}, resources={"nic": 10},
                     profiles=(RateProfile("bad", ("a", "b"), {"a": 8, "b": 8}, resources=("nic",), measured_at="2026-11-03T00:00:00Z"),))
    executable = Executable(tuple(WorkOperation(name, "send", "wire", 10, flow_id=name, resources=("nic",)) for name in ("a", "b")))
    with pytest.raises(SemanticError, match="capacity"):
        predict(executable, cost)


def test_measured_profile_requires_its_original_measurement_time():
    from vericcl.performance.profiles import RateProfile
    with pytest.raises(SemanticError, match="measurement time"):
        RateProfile("missing-time", ("a",), {"a": 1})


def test_topology_loader_keeps_v5_beta_in_microseconds_per_byte(tmp_path):
    from tests.unit.input.v5_support import write_request
    from vericcl.input.loader import load_request
    from vericcl.topology.loader import load_topology
    from vericcl.topology.model import LinkKey
    from vericcl.topology.performance import transfer_duration_us
    inputs = load_request(write_request(tmp_path))
    topology = load_topology(inputs)
    edge = topology.links[LinkKey(0, 1)]
    assert transfer_duration_us(edge, 1024, 1) == pytest.approx(11.24)
    assert transfer_duration_us(edge, 2048, 1) == pytest.approx(21.48)


def test_canonical_shared_capacity_is_not_reinterpreted_as_fixed_latency(tmp_path):
    from tests.unit.input.v5_support import request_documents, write_request
    from vericcl.input.loader import load_request
    from vericcl.topology.loader import load_topology
    from vericcl.topology.performance import transfer_duration_us
    from vericcl.topology.model import LinkKey
    documents = request_documents()
    edge = documents['topology']['directed_links'][0]
    edge['resources'] = ['nic']
    documents['topology']['shared_resources'] = [{'id': 'nic', 'capacity_bytes_per_us': 50}]
    topology = load_topology(load_request(write_request(tmp_path, documents)))
    resource = topology.shared_resources['nic']
    assert resource.member_links == (LinkKey(0, 1),)
    assert resource.capacity_bytes_per_us == 50
    assert resource.performance.alpha_us == 0
    assert topology.links[LinkKey(0, 1)].performance.beta_us_per_byte == .01


def test_profile_update_records_actual_parameter_differences():
    from vericcl.performance.model import CostModel
    from vericcl.performance.profiles import RateProfile
    initial = CostModel('m0', {})
    profile = RateProfile('same-id', ('a',), {'a': 2}, source='estimated')
    updated, _ = initial.with_updates('m1', profiles=(profile,))
    _, changes = updated.with_updates('m2', profiles=(RateProfile('same-id', ('a',), {'a': 3}, source='estimated'),))
    assert changes[0]['before'][0]['rates']['a'] == 2
    assert changes[0]['after'][0]['rates']['a'] == 3


def test_new_executable_uses_progress_model_through_verification_entry():
    from vericcl.performance.events import Executable, WorkOperation
    from vericcl.performance.model import CostModel, ServiceCost
    from vericcl.verification.simulator import simulate_schedule
    result = simulate_schedule(Executable((WorkOperation('a', 'copy', 'c', 10),)),
                               CostModel('m0', {'c': (ServiceCost(1, .1),)}))
    assert result.t_pred_us == 2
