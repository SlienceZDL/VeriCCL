import pytest

from tests.e2e._support import (
    assert_exact_tiny_buffers,
    assert_semantic_outputs,
    assert_xml_contract,
    solve_public_cli,
)
from tests.e2e.test_six_collectives import OPERATORS


pytestmark = pytest.mark.phase07


@pytest.mark.parametrize("operator", OPERATORS)
@pytest.mark.parametrize("inplace", (False, True))
def test_buffer_addresses_preserve_inplace_mode(tmp_path, operator, inplace):
    result = solve_public_cli(tmp_path, operator, inplace=inplace)

    assert result["xml"].attrib["inplace"] == str(int(inplace))
    assert_exact_tiny_buffers(result, operator, inplace)
    assert_semantic_outputs(result, operator)
    assert_xml_contract(result)
