from __future__ import annotations

import time

import pytest

from tests.e2e._support import (
    assert_no_global_stage_barrier,
    assert_semantic_outputs,
    assert_validation_report,
    assert_xml_contract,
    solve_public_cli,
    transfer_pairs,
    write_inputs,
    write_multi_rail_topology,
)
from tests.gurobi.helpers import require_gurobi_license
from vericcl.input.loader import resolve_inputs
from vericcl.planner.build import build_plan
from vericcl.solver.cache import CandidateCache
from vericcl.solver.model import SolveRequest, SolveStatus
from vericcl.solver.orchestrator import solve
from vericcl.topology.loader import load_topology


pytestmark = pytest.mark.phase07


def test_one_gateway_allgather_preserves_complete_real_slice_semantics(tmp_path):
    result = solve_public_cli(
        tmp_path,
        "allgather",
        topology_name="two_node_gateway.json",
        total_size_bytes=8192,
        slice_size_bytes=1024,
        hierarchy=True,
    )
    schedule = result["sidecar"]["schedule"]
    pairs = transfer_pairs(schedule["transfers"])

    assert result["report"]["hierarchy_plan"]["planning_mode"] == (
        "gateway_allgather"
    )
    assert (1, 5) not in pairs and (5, 1) not in pairs
    assert {(0, 4), (4, 0)} <= pairs
    assert_no_global_stage_barrier(result)
    assert_semantic_outputs(result, "allgather")
    assert_validation_report(result)
    assert_xml_contract(result)


def test_four_rail_allgather_partitions_real_slices_without_fake_links(tmp_path):
    topology_path = write_multi_rail_topology(tmp_path / "four-rail.json")
    result = solve_public_cli(
        tmp_path,
        "allgather",
        topology_path=topology_path,
        total_size_bytes=8192,
        slice_size_bytes=1024,
        hierarchy=True,
    )
    transfers = result["sidecar"]["schedule"]["transfers"]
    node_zero = {0, 1, 2, 3}
    rails = {(0, 4): 0, (1, 5): 1, (2, 6): 2, (3, 7): 3}
    directed_rails = rails | {
        (right, left): rail for (left, right), rail in rails.items()
    }
    cross_node = [
        transfer
        for transfer in transfers
        if (transfer["src_rank"] in node_zero)
        != (transfer["dst_rank"] in node_zero)
    ]

    assert result["report"]["hierarchy_plan"]["planning_mode"] == (
        "gateway_allgather"
    )
    assert set(transfer_pairs(cross_node)) == set(directed_rails)
    for transfer in cross_node:
        rail = directed_rails[(transfer["src_rank"], transfer["dst_rank"])]
        assert transfer["member_slice_ids"]
        assert all(slice_id % 4 == rail for slice_id in transfer["member_slice_ids"])
    assert_no_global_stage_barrier(result)
    assert_semantic_outputs(result, "allgather")
    assert_validation_report(result)
    assert_xml_contract(result)


@pytest.mark.gurobi
def test_one_gateway_scalable_allgather_validates_and_lowers_end_to_end(tmp_path):
    require_gurobi_license()
    result = solve_public_cli(
        tmp_path,
        "allgather",
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
    assert report["search_diagnostics"]["template_count"] == 21
    assert report["search_diagnostics"]["route_model_count"] == 21
    assert_no_global_stage_barrier(result)
    assert_semantic_outputs(result, "allgather")
    assert_validation_report(result)
    assert_xml_contract(result)


@pytest.mark.gurobi
def test_gateway_route_model_count_is_constant_as_real_slices_scale(tmp_path):
    require_gurobi_license()
    observations = []
    started = time.monotonic()

    for slice_count in (8, 16, 64, 128):
        inputs_dir = tmp_path / "scale-{}".format(slice_count)
        inputs_dir.mkdir()
        topology_path, sketch_path, atom_path = write_inputs(
            inputs_dir,
            "allgather",
            topology_name="two_node_gateway.json",
            total_size_bytes=slice_count * 1024,
            slice_size_bytes=1024,
            max_channels=1,
            hierarchy=True,
            constructive_trees=False,
            milp=True,
            max_parallel_models=4,
            max_threads_per_model=1,
            total_solve_timeout_s=300,
            per_model_timeout_s=60,
        )
        inputs = resolve_inputs(topology_path, sketch_path, atom_path)
        topology = load_topology(inputs)
        request = SolveRequest(
            inputs=inputs,
            topology=topology,
            plan=build_plan(inputs, topology),
            solver_version="task9-gurobi",
            model_version="task9-route-model",
            environment_signature="task9-node2",
        )

        result = solve(request, cache=CandidateCache())

        assert result.status is SolveStatus.FEASIBLE
        candidate = result.selected_candidate
        assert candidate is not None and candidate.global_schedule is not None
        assert candidate.search_space_restricted
        assert not candidate.proven_optimal
        assert result.diagnostics.fallback_member_model_count == 0
        observations.append(
            (
                result.diagnostics.requested_problem_count,
                result.diagnostics.template_count,
                result.diagnostics.template_member_count,
                result.diagnostics.route_model_count,
                len(candidate.global_schedule.transfers),
            )
        )

    assert {item[0] for item in observations} == {5}
    assert {item[1] for item in observations} == {21}
    assert {item[3] for item in observations} == {21}
    assert [item[2] for item in observations] == [240, 480, 1920, 3840]
    assert [item[4] for item in observations] == [448, 896, 3584, 7168]
    assert time.monotonic() - started < 300.0
