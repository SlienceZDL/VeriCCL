import hashlib

import pytest

from tests.e2e._support import (
    canonical_report_sections,
    solve_public_cli,
)


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
