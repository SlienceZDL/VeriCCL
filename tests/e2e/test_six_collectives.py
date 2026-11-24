import pytest

from tests.e2e._support import OPERATORS, assert_end_to_end, documents


pytestmark = pytest.mark.phase07


@pytest.mark.parametrize('operator', OPERATORS)
def test_public_solve_emits_executable_semantically_exact_collective(solved, operator):
    run = solved(documents(operator))

    assert run.xml.attrib['coll'] == operator
    assert run.sidecar['request_contract']['operator'] == operator
    assert_end_to_end(run)
