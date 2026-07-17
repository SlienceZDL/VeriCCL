import pytest

from tests.e2e._support import (
    assert_semantic_outputs,
    assert_validation_report,
    assert_xml_contract,
    solve_public_cli,
    transfer_pairs,
)


pytestmark = pytest.mark.phase07


def test_gateway_hierarchy_composes_exact_global_allreduce(tmp_path):
    result = solve_public_cli(
        tmp_path,
        "allreduce",
        topology_name="two_node_gateway.json",
        total_size_bytes=8192,
        slice_size_bytes=1024,
        hierarchy=True,
    )
    schedule = result["sidecar"]["schedule"]
    transfers = schedule["transfers"]
    pairs = transfer_pairs(transfers)

    assert (1, 5) not in pairs and (5, 1) not in pairs
    assert any(
        transfer["kind"] == "REDUCE"
        and transfer["dst_rank"] == 0
        and transfer["src_rank"] in {1, 2, 3}
        for transfer in transfers
    )
    assert any(
        transfer["kind"] == "REDUCE"
        and transfer["dst_rank"] == 4
        and transfer["src_rank"] in {5, 6, 7}
        for transfer in transfers
    )
    inter = [
        transfer
        for transfer in transfers
        if {transfer["src_rank"], transfer["dst_rank"]} == {0, 4}
    ]
    assert {transfer["kind"] for transfer in inter} == {"SEND", "REDUCE"}
    assert any(
        transfer["kind"] == "SEND"
        and transfer["src_rank"] == 0
        and transfer["dst_rank"] in {1, 2, 3}
        for transfer in transfers
    )
    assert any(
        transfer["kind"] == "SEND"
        and transfer["src_rank"] == 4
        and transfer["dst_rank"] in {5, 6, 7}
        for transfer in transfers
    )
    hierarchy = result["report"]["hierarchy_plan"]
    assert [node["stage_id"] for node in hierarchy["nodes"]] == [
        0,
        0,
        1,
        2,
        3,
        3,
    ]
    assert schedule["metadata"].get("stage_barrier") is None
    by_stage = {
        stage: [
            transfer
            for transfer in transfers
            if transfer["stage_id"] == stage
        ]
        for stage in range(4)
    }
    assert all(
        min(transfer["st_time"] for transfer in by_stage[stage + 1])
        < max(transfer["ed_time"] for transfer in by_stage[stage])
        for stage in range(3)
    )
    assert_semantic_outputs(result, "allreduce")
    assert_validation_report(result)
    assert_xml_contract(result)
