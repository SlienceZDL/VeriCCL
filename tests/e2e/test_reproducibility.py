import hashlib

import pytest

from tests.e2e._support import (
    canonical_report_sections,
    solve_public_cli,
)
from tests.gurobi.helpers import require_gurobi_license


pytestmark = pytest.mark.phase07


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_seed_zero_pure_software_runs_are_canonically_reproducible(tmp_path):
    first = solve_public_cli(tmp_path / "first", "allreduce", run_id="first")
    second = solve_public_cli(
        tmp_path / "second",
        "allreduce",
        run_id="second",
    )

    assert first["sidecar"] == second["sidecar"]
    assert canonical_report_sections(first["report"]) == (
        canonical_report_sections(second["report"])
    )
    assert _digest(first["xml_path"]) == _digest(second["xml_path"])
    assert first["report"]["xml_sha256"] == second["report"]["xml_sha256"]
    reproducibility = first["report"]["reproducibility"]
    assert reproducibility["solver_seed"] == 0
    assert reproducibility["deterministic_artifacts"] is True
    assert set(reproducibility["limits"]) == {
        "environment_signature",
        "hardware_measurement",
        "parallel_solver_execution",
        "solver_version",
    }


@pytest.mark.gurobi
def test_scalable_route_composition_is_reproducible_except_phase_timings(tmp_path):
    require_gurobi_license()
    arguments = {
        "total_size_bytes": 4096,
        "slice_size_bytes": 1024,
        "max_channels": 1,
        "constructive_trees": False,
        "milp": True,
    }
    first = solve_public_cli(
        tmp_path / "first-scalable",
        "allgather",
        run_id="first-scalable",
        **arguments,
    )
    second = solve_public_cli(
        tmp_path / "second-scalable",
        "allgather",
        run_id="second-scalable",
        **arguments,
    )

    assert first["sidecar"]["schedule"] == second["sidecar"]["schedule"]
    assert _digest(first["xml_path"]) == _digest(second["xml_path"])
    assert first["report"]["effective_solving"] == (
        second["report"]["effective_solving"]
    )
    structural_fields = {
        "requested_problem_count",
        "template_count",
        "template_member_count",
        "route_model_count",
        "fallback_member_model_count",
        "maximum_variable_count",
        "maximum_constraint_count",
        "maximum_general_constraint_count",
    }
    assert {
        key: value
        for key, value in first["report"]["search_diagnostics"].items()
        if key in structural_fields
    } == {
        key: value
        for key, value in second["report"]["search_diagnostics"].items()
        if key in structural_fields
    }
    assert first["report"]["effective_solving"] == {
        "global_proven_optimal": False,
        "planning_mode": "direct",
        "requested_gap_satisfied": False,
        "requested_hierarchy": False,
        "restricted_template_composition": True,
        "search_space_restricted": True,
        "selected_best": True,
        "solver_strategy": "scalable_template_routing",
    }
