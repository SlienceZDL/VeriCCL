import pytest

from tests.e2e._support import OPERATORS, assert_end_to_end, documents, steps


pytestmark = pytest.mark.phase07


@pytest.mark.parametrize('operator', OPERATORS)
def test_inplace_mode_is_lowered_into_aliased_buffers(solved, operator):
    runs = {inplace: solved(documents(operator, inplace=inplace)) for inplace in (False, True)}

    for inplace, run in runs.items():
        assert run.xml.attrib['inplace'] == str(int(inplace))
        assert run.sidecar['request_contract']['inplace'] is inplace
        # The replay inside aliases i/o buffers for in-place XML, so a schedule
        # that overwrote a still-needed input would fail here.
        assert_end_to_end(run)
    copies = {inplace: len(steps(run, 'cpy')) for inplace, run in runs.items()}
    assert copies[True] <= copies[False]
    if operator in {'broadcast', 'allgather'}:
        # Out-of-place must copy the local block into the output buffer;
        # in-place already holds it at the aliased output offset.
        assert copies[False] > 0 and copies[True] == 0
