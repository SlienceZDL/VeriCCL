import pytest

from tests.e2e._support import (
    assert_semantic_outputs,
    solve_public_cli,
)


pytestmark = pytest.mark.phase07


def test_runtime_incompatible_schedule_remains_offline_valid_candidate(tmp_path):
    result = solve_public_cli(
        tmp_path,
        "broadcast",
        total_size_bytes=257 * 1024,
        slice_size_bytes=1024,
        max_channels=1,
    )
    report = result["report"]

    assert result["xml_path"].name.endswith("_final.candidate.xml")
    assert report["runtime_compatible"] is False
    assert report["validation"]["runtime"]["status"] == "warning"
    assert report["validation"]["semantic"]["status"] == "valid"
    assert report["validation"]["bdd"]["status"] == "valid"
    assert report["validation"]["simulation"]["status"] == "valid"
    assert "steps_per_tb" in {
        issue["code"]
        for issue in report["validation"]["runtime"]["evidence"]["issues"]
    }
    recommendations = {
        recommendation["kind"]: recommendation
        for recommendation in report["runtime_recommendations"]
    }
    assert recommendations["increase_channels"]["parameters"] == {
        "max_channels": 2
    }
    assert (
        recommendations["increase_slice_size"]["parameters"][
            "slice_size_bytes"
        ]
        > 1024
    )
    assert_semantic_outputs(result, "broadcast")


def test_candidate_report_preserves_template_backend_restrictions(tmp_path):
    result = solve_public_cli(
        tmp_path,
        "broadcast",
        total_size_bytes=2048,
        slice_size_bytes=1024,
        max_channels=1,
    )

    assert result["report"]["restrictions"] == [
        "independent_node_composition",
        "template_route_composition",
    ]
    assert result["sidecar"]["candidate"]["restrictions"] == (
        result["report"]["restrictions"]
    )
    assert result["report"]["search_space_restricted"] is True
    assert result["report"]["proven_optimal"] is False
