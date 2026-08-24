import pytest

from tests.e2e._support import (
    assert_cross_stage_accumulator_dependencies,
    assert_no_global_stage_barrier,
    assert_reduction_atoms,
    assert_semantic_outputs,
    assert_validation_report,
    assert_xml_contract,
    solve_public_cli,
    transfer_pairs,
    write_multi_rail_topology,
)
from tests.gurobi.helpers import require_gurobi_license


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
    assert_reduction_atoms(result)
    assert_cross_stage_accumulator_dependencies(result)
    assert_validation_report(result)
    assert_xml_contract(result)


def test_four_rail_allreduce_uses_only_declared_bidirectional_gateways(tmp_path):
    topology_path = write_multi_rail_topology(tmp_path / "four-rail.json")
    result = solve_public_cli(
        tmp_path,
        "allreduce",
        topology_path=topology_path,
        total_size_bytes=8192,
        slice_size_bytes=1024,
        hierarchy=True,
    )
    transfers = result["sidecar"]["schedule"]["transfers"]
    node_zero = {0, 1, 2, 3}
    declared = {
        (0, 4),
        (4, 0),
        (1, 5),
        (5, 1),
        (2, 6),
        (6, 2),
        (3, 7),
        (7, 3),
    }
    cross_node = [
        transfer
        for transfer in transfers
        if (transfer["src_rank"] in node_zero)
        != (transfer["dst_rank"] in node_zero)
    ]

    assert result["report"]["hierarchy_plan"]["planning_mode"] == (
        "gateway_allreduce"
    )
    assert cross_node
    assert transfer_pairs(cross_node) <= declared
    assert_no_global_stage_barrier(result)
    assert_semantic_outputs(result, "allreduce")
    assert_reduction_atoms(result)
    assert_cross_stage_accumulator_dependencies(result)
    assert_validation_report(result)
    assert_xml_contract(result)


@pytest.mark.gurobi
def test_one_gateway_scalable_allreduce_validates_and_lowers_end_to_end(tmp_path):
    require_gurobi_license()
    result = solve_public_cli(
        tmp_path,
        "allreduce",
        topology_name="two_node_gateway.json",
        total_size_bytes=8192,
        slice_size_bytes=1024,
        max_channels=1,
        hierarchy=True,
        constructive_trees=False,
        milp=True,
        max_parallel_models=4,
        max_threads_per_model=1,
    )
    report = result["report"]

    assert report["effective_solving"]["solver_strategy"] == (
        "scalable_template_routing"
    )
    assert report["effective_solving"]["restricted_template_composition"] is True
    assert report["effective_solving"]["global_proven_optimal"] is False
    assert report["search_diagnostics"]["template_count"] > 0
    assert report["search_diagnostics"]["route_model_count"] == (
        report["search_diagnostics"]["template_count"]
    )
    assert_no_global_stage_barrier(result)
    assert_semantic_outputs(result, "allreduce")
    assert_reduction_atoms(result)
    assert_cross_stage_accumulator_dependencies(result)
    assert_validation_report(result)
    assert_xml_contract(result)
