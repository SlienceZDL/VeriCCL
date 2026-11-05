import pytest

from vericcl.errors import InputValidationError
from vericcl.input.validation import validate_collective
from vericcl.semantics.collective import CollectiveKind, CollectiveSpec


pytestmark = pytest.mark.phase01


@pytest.mark.parametrize("kind", [CollectiveKind.BROADCAST, CollectiveKind.REDUCE])
def test_rooted_collectives_require_root(kind):
    spec = CollectiveSpec(kind=kind, datatype="float32", reduction_op="sum")

    with pytest.raises(InputValidationError, match="root"):
        validate_collective(spec, rank_count=2, slice_count=8)


@pytest.mark.parametrize("root", [-1, 2])
def test_root_must_be_in_range(root):
    spec = CollectiveSpec(
        kind=CollectiveKind.BROADCAST,
        datatype="float32",
        root=root,
    )

    with pytest.raises(InputValidationError, match="root"):
        validate_collective(spec, rank_count=2, slice_count=8)


def test_rootless_collective_rejects_root():
    spec = CollectiveSpec(
        kind=CollectiveKind.ALL_GATHER,
        datatype="float32",
        root=0,
    )

    with pytest.raises(InputValidationError, match="must not define root"):
        validate_collective(spec, rank_count=2, slice_count=8)


@pytest.mark.parametrize(
    "kind",
    [CollectiveKind.REDUCE, CollectiveKind.ALL_REDUCE, CollectiveKind.REDUCE_SCATTER],
)
def test_reduction_collectives_require_reduction_operation(kind):
    root = 0 if kind is CollectiveKind.REDUCE else None
    spec = CollectiveSpec(kind=kind, datatype="float32", root=root)

    with pytest.raises(InputValidationError, match="reduction_op"):
        validate_collective(spec, rank_count=2, slice_count=8)


def test_unknown_reduction_operation_is_rejected():
    spec = CollectiveSpec(
        kind=CollectiveKind.ALL_REDUCE,
        datatype="float32",
        reduction_op="xor",
    )

    with pytest.raises(InputValidationError, match="unsupported reduction_op"):
        validate_collective(spec, rank_count=2, slice_count=8)


@pytest.mark.parametrize(
    "kind",
    [CollectiveKind.ALL_TO_ALL, CollectiveKind.REDUCE_SCATTER],
)
def test_partitioning_collectives_require_rank_divisibility(kind):
    reduction_op = "sum" if kind is CollectiveKind.REDUCE_SCATTER else None
    spec = CollectiveSpec(kind=kind, datatype="float32", reduction_op=reduction_op)

    with pytest.raises(InputValidationError, match="slice count must be divisible"):
        validate_collective(spec, rank_count=4, slice_count=6)


def test_valid_collective_is_accepted():
    spec = CollectiveSpec(
        kind=CollectiveKind.ALL_REDUCE,
        datatype="float32",
        reduction_op="sum",
    )

    validate_collective(spec, rank_count=2, slice_count=8)


@pytest.mark.parametrize(
    "rank_count,slice_count,datatype",
    [
        (True, 8, "float32"),
        (0, 8, "float32"),
        (2, True, "float32"),
        (2, 0, "float32"),
        (2, 8, ""),
    ],
)
def test_invalid_collective_geometry_is_rejected(
    rank_count,
    slice_count,
    datatype,
):
    spec = CollectiveSpec(kind=CollectiveKind.ALL_GATHER, datatype=datatype)

    with pytest.raises(InputValidationError):
        validate_collective(spec, rank_count, slice_count)


def test_non_reduction_collective_rejects_reduction_operation():
    spec = CollectiveSpec(
        kind=CollectiveKind.ALL_GATHER,
        datatype="float32",
        reduction_op="sum",
    )

    with pytest.raises(InputValidationError, match="must not define reduction_op"):
        validate_collective(spec, rank_count=2, slice_count=8)
