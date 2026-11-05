import pytest

from vericcl.semantics.slice import logical_slice_index, source_rank


pytestmark = pytest.mark.phase01


def test_slice_identity_uses_global_slice_count():
    assert source_rank(7, 4) == 1
    assert logical_slice_index(7, 4) == 3


@pytest.mark.parametrize("slice_id", [-1, True])
def test_slice_identity_rejects_invalid_slice_id(slice_id):
    with pytest.raises(ValueError, match="slice_id"):
        source_rank(slice_id, 4)


@pytest.mark.parametrize("slice_count", [0, -1, True])
def test_slice_identity_rejects_invalid_global_slice_count(slice_count):
    with pytest.raises(ValueError, match="slice_count"):
        logical_slice_index(0, slice_count)
