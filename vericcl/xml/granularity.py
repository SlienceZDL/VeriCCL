from vericcl.errors import SemanticError


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SemanticError("{} must be a positive integer".format(field))
    return value


def verify_atom_granularity(
    runtime_count: int,
    size_multiplier: int,
    datatype_size_bytes: int,
    nchunks_per_loop: int,
    slice_size_bytes: int,
    nccl_buffsize_bytes: int,
) -> None:
    runtime_count = _positive_integer(runtime_count, "runtime_count")
    size_multiplier = _positive_integer(size_multiplier, "size_multiplier")
    datatype_size_bytes = _positive_integer(
        datatype_size_bytes,
        "datatype_size_bytes",
    )
    nchunks_per_loop = _positive_integer(
        nchunks_per_loop,
        "nchunks_per_loop",
    )
    slice_size_bytes = _positive_integer(
        slice_size_bytes,
        "slice_size_bytes",
    )
    nccl_buffsize_bytes = _positive_integer(
        nccl_buffsize_bytes,
        "nccl_buffsize_bytes",
    )
    runtime_bytes = runtime_count * size_multiplier * datatype_size_bytes
    if runtime_bytes % nchunks_per_loop:
        raise SemanticError("runtime bytes must be divisible by nchunks_per_loop")
    if runtime_bytes // nchunks_per_loop != slice_size_bytes:
        raise SemanticError("runtime chunk size does not equal the slice size")
    if nccl_buffsize_bytes != 2 * slice_size_bytes:
        raise SemanticError("NCCL_BUFFSIZE must equal two slice sizes")
