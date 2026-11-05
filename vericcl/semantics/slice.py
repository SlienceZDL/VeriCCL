def _validate_slice_identity_input(value: object, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("{} must be an integer".format(field))
    if value < minimum:
        raise ValueError("{} must be at least {}".format(field, minimum))
    return value


def source_rank(slice_id: int, slice_count: int) -> int:
    normalized_slice_id = _validate_slice_identity_input(slice_id, "slice_id", 0)
    normalized_slice_count = _validate_slice_identity_input(
        slice_count,
        "slice_count",
        1,
    )
    return normalized_slice_id // normalized_slice_count


def logical_slice_index(slice_id: int, slice_count: int) -> int:
    normalized_slice_id = _validate_slice_identity_input(slice_id, "slice_id", 0)
    normalized_slice_count = _validate_slice_identity_input(
        slice_count,
        "slice_count",
        1,
    )
    return normalized_slice_id % normalized_slice_count
