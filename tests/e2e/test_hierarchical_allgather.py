import pytest

from vericcl.artifacts.writer import read_schedule_sidecar
from vericcl.input.loader import resolve_inputs
from vericcl.topology.loader import load_topology
from vericcl.verification.bdd_flow import analyze_flow_congestion
from vericcl.verification.flow_index import build_flow_index
from vericcl.verification.model import ValidationStatus

from tests.e2e._support import (
    assert_semantic_outputs,
    assert_validation_report,
    assert_xml_contract,
    solve_public_cli,
    transfer_pairs,
)


pytestmark = pytest.mark.phase07


def test_gateway_hierarchy_composes_exact_global_allgather(tmp_path):
    result = solve_public_cli(
        tmp_path,
        "allgather",
        topology_name="two_node_gateway.json",
        total_size_bytes=8192,
        slice_size_bytes=1024,
        hierarchy=True,
    )
    schedule = result["sidecar"]["schedule"]
    transfers = schedule["transfers"]

    assert result["report"]["planning_mode"] == "gateway_allgather"
    assert "template_route_composition" in result["report"]["restrictions"]
    assert (1, 5) not in transfer_pairs(transfers)
    assert (5, 1) not in transfer_pairs(transfers)
    assert all(transfer["kind"] == "SEND" for transfer in transfers)
    assert {transfer["stage_id"] for transfer in transfers} == {0, 1, 2}
    assert {
        step.attrib["type"]
        for step in result["xml"].xpath("./gpu/tb/step")
    } == {"cpy", "r", "s"}
    assert all(
        (int(tb.attrib["send"]) >= 0) != (int(tb.attrib["recv"]) >= 0)
        for tb in result["xml"].xpath("./gpu/tb")
        if tb.xpath("./step[@type='s' or @type='r' or @type='rrc']")
    )
    assert schedule["metadata"].get("stage_barrier") is None
    by_stage = {
        stage: [
            transfer
            for transfer in transfers
            if transfer["stage_id"] == stage
        ]
        for stage in range(3)
    }
    assert all(
        min(transfer["st_time"] for transfer in by_stage[stage + 1])
        < max(transfer["ed_time"] for transfer in by_stage[stage])
        for stage in range(2)
    )
    for gpu in result["xml"].xpath("./gpu"):
        threadblocks = {
            int(tb.attrib["id"]): tb for tb in gpu.xpath("./tb")
        }
        for tb in threadblocks.values():
            for step in tb.xpath("./step"):
                depid = int(step.attrib["depid"])
                deps = int(step.attrib["deps"])
                if depid < 0:
                    assert deps == -1
                    continue
                assert deps >= 0
                assert deps < len(threadblocks[depid].xpath("./step"))
    assert_semantic_outputs(result, "allgather")
    assert_validation_report(result)
    assert_xml_contract(result)


def test_gateway_template_sidecar_builds_instantiated_bdd_flows(tmp_path):
    result = solve_public_cli(
        tmp_path,
        "allgather",
        topology_name="two_node_gateway.json",
        total_size_bytes=8192,
        slice_size_bytes=1024,
        hierarchy=True,
    )
    sidecar = read_schedule_sidecar(result["sidecar_path"])
    inputs = resolve_inputs(*result["input_paths"])
    topology = load_topology(inputs)

    index = build_flow_index(sidecar.schedule)
    analysis = analyze_flow_congestion(
        sidecar.schedule,
        topology,
        inputs,
    )

    assert index.flows
    assert analysis.status is ValidationStatus.VALID
    assert result["report"]["validation"]["bdd"]["status"] == "valid"
    transfer_ids = {
        transfer.transfer_id for transfer in sidecar.schedule.transfers
    }
    for flow in index.flows:
        assert flow.transfer_ids
        assert set(flow.transfer_ids) <= transfer_ids
        assert flow.logical_slice_index == (
            next(iter(flow.member_slice_ids)) % sidecar.schedule.slice_count
        )
        assert flow.root_rank == flow.ranks[0]
        assert flow.leaf_rank == flow.ranks[-1]
        assert "template" not in flow.flow_id
        assert "template" not in flow.demand_id
