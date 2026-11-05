from vericcl.errors import InputValidationError
from vericcl.semantics.collective import CollectiveKind, CollectiveSpec


SUPPORTED_REDUCTION_OPERATIONS = frozenset({"avg", "max", "min", "prod", "sum"})
_ROOTED_COLLECTIVES = frozenset(
    {
        CollectiveKind.BROADCAST,
        CollectiveKind.REDUCE,
        CollectiveKind.SCATTER,
        CollectiveKind.GATHER,
    }
)
_REDUCTION_COLLECTIVES = frozenset(
    {
        CollectiveKind.REDUCE,
        CollectiveKind.ALL_REDUCE,
        CollectiveKind.REDUCE_SCATTER,
    }
)
_PARTITIONING_COLLECTIVES = frozenset(
    {
        CollectiveKind.SCATTER,
        CollectiveKind.ALL_TO_ALL,
        CollectiveKind.REDUCE_SCATTER,
    }
)


def validate_collective(
    spec: CollectiveSpec,
    rank_count: int,
    slice_count: int,
) -> None:
    if isinstance(rank_count, bool) or not isinstance(rank_count, int):
        raise InputValidationError("rank count must be an integer")
    if rank_count <= 0:
        raise InputValidationError("rank count must be positive")
    if isinstance(slice_count, bool) or not isinstance(slice_count, int):
        raise InputValidationError("slice count must be an integer")
    if slice_count <= 0:
        raise InputValidationError("slice count must be positive")
    if not isinstance(spec.datatype, str) or not spec.datatype:
        raise InputValidationError("datatype must be a non-empty string")

    if spec.kind in _ROOTED_COLLECTIVES:
        if isinstance(spec.root, bool) or not isinstance(spec.root, int):
            raise InputValidationError(
                "{} requires an integer root".format(spec.kind.value)
            )
        if spec.root < 0 or spec.root >= rank_count:
            raise InputValidationError(
                "root must be in the range [0, {})".format(rank_count)
            )
    elif spec.root is not None:
        raise InputValidationError(
            "{} must not define root".format(spec.kind.value)
        )

    if spec.kind in _REDUCTION_COLLECTIVES:
        if not isinstance(spec.reduction_op, str) or not spec.reduction_op:
            raise InputValidationError(
                "{} requires reduction_op".format(spec.kind.value)
            )
        if spec.reduction_op not in SUPPORTED_REDUCTION_OPERATIONS:
            raise InputValidationError(
                "unsupported reduction_op: {}".format(spec.reduction_op)
            )
    elif spec.reduction_op is not None:
        raise InputValidationError(
            "{} must not define reduction_op".format(spec.kind.value)
        )

    if spec.kind in _PARTITIONING_COLLECTIVES and slice_count % rank_count != 0:
        raise InputValidationError(
            "slice count must be divisible by rank count for {}".format(
                spec.kind.value
            )
        )
