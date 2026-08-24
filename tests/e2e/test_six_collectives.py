import pytest

from tests.e2e._support import (
    assert_exact_tiny_buffers,
    assert_semantic_outputs,
    assert_validation_report,
    assert_xml_contract,
    solve_public_cli,
)


pytestmark = pytest.mark.phase07


OPERATORS = (
    "broadcast",
    "reduce",
    "allgather",
    "allreduce",
    "alltoall",
    "reducescatter",
)


@pytest.mark.parametrize("operator", OPERATORS)
def test_public_cli_emits_semantic_valid_direct_collective(tmp_path, operator):
    result = solve_public_cli(tmp_path, operator)

    xml_operator = "reduce_scatter" if operator == "reducescatter" else operator
    assert result["xml"].attrib["coll"] == xml_operator
    assert_exact_tiny_buffers(result, operator, False)
    assert_semantic_outputs(result, operator)
    assert_validation_report(result)
    assert_xml_contract(result)
    report = result["report"]
    assert report["hierarchy_plan"]["planning_mode"] == "direct"
    assert report["effective_solving"]["solver_strategy"] == "constructive"
    assert report["effective_solving"]["global_proven_optimal"] is False
    assert report["search_diagnostics"]["route_model_count"] == 0
