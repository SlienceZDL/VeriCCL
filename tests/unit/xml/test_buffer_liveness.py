from dataclasses import replace

import pytest

from vericcl.errors import SemanticError
from vericcl.semantics.collective import CollectiveKind
from vericcl.xml.buffers import build_buffer_plan
from vericcl.xml.liveness import verify_buffer_liveness
from vericcl.xml.model import AggregateValue, PhysicalRef, RawValue

from tests.unit.xml.helpers import (
    concurrent_reduce_star_schedule,
    reduce_chain_schedule,
    resolved,
)


pytestmark = pytest.mark.phase04


def test_non_inplace_reduce_initializes_each_accumulator_with_local_copy():
    schedule = reduce_chain_schedule()
    inputs = resolved(CollectiveKind.REDUCE, ranks=3, slices=1)

    plan = build_buffer_plan(schedule, inputs)

    assert [copy.reason for copy in plan.local_copies] == [
        "initialize reduction accumulator",
        "initialize reduction accumulator",
    ]
    assert plan.transfer_dst_refs["reduce-a0-first"].buffer == "s"
    assert plan.transfer_dst_refs["reduce-a0-second"].buffer == "o"
    verify_buffer_liveness(schedule, plan, inputs)


def test_inplace_reduce_uses_alias_only_at_root_and_scratch_at_intermediate():
    schedule = reduce_chain_schedule()
    inputs = resolved(
        CollectiveKind.REDUCE,
        ranks=3,
        slices=1,
        inplace=True,
    )

    plan = build_buffer_plan(schedule, inputs)

    assert [copy.reason for copy in plan.local_copies] == [
        "initialize reduction accumulator"
    ]
    assert plan.transfer_dst_refs["reduce-a0-first"].buffer == "s"
    assert plan.transfer_dst_refs["reduce-a0-second"].buffer == "o"
    verify_buffer_liveness(schedule, plan, inputs)


def test_concurrent_reductions_share_one_accumulator_in_stable_order():
    schedule = concurrent_reduce_star_schedule()
    inputs = resolved(CollectiveKind.REDUCE, ranks=4, slices=1)

    plan = build_buffer_plan(schedule, inputs)

    assert plan.transfer_effective_times == {
        "reduce-star-1": (0.0, 1.0),
        "reduce-star-2": (1.0, 2.0),
        "reduce-star-3": (2.0, 3.0),
    }
    assert [
        sorted(value.contributors)
        if isinstance(value, AggregateValue)
        else [value.slice_id]
        for value in plan.transfer_accumulator_values.values()
    ] == [[0], [0, 1], [0, 1, 2]]
    verify_buffer_liveness(schedule, plan, inputs)


def test_liveness_rejects_overlapping_values_in_one_scratch_slot():
    schedule = reduce_chain_schedule()
    inputs = resolved(CollectiveKind.REDUCE, ranks=3, slices=1)
    plan = build_buffer_plan(schedule, inputs)
    aggregate = next(
        value
        for value in plan.value_locations
        if isinstance(value, AggregateValue)
        and any(ref.buffer == "s" for ref in plan.value_locations[value])
    )
    raw = RawValue(0)
    bad_locations = dict(plan.value_locations)
    bad_locations[raw] = bad_locations[raw] + (
        PhysicalRef(
            rank=1,
            buffer="s",
            offset=0,
            valid_from=0.0,
            valid_until=10.0,
        ),
    )
    bad_plan = replace(plan, value_locations=bad_locations)

    with pytest.raises(SemanticError, match="live values share a physical location"):
        verify_buffer_liveness(schedule, bad_plan, inputs)


def test_liveness_rejects_out_of_bounds_transfer_reference():
    schedule = reduce_chain_schedule()
    inputs = resolved(CollectiveKind.REDUCE, ranks=3, slices=1)
    plan = build_buffer_plan(schedule, inputs)
    bad_refs = dict(plan.transfer_src_refs)
    bad_refs["reduce-a0-first"] = replace(
        bad_refs["reduce-a0-first"],
        offset=plan.i_chunks[2],
    )

    with pytest.raises(SemanticError, match="buffer offset is outside declared chunks"):
        verify_buffer_liveness(
            schedule,
            replace(plan, transfer_src_refs=bad_refs),
            inputs,
        )


@pytest.mark.parametrize(
    "field",
    [
        "transfer_src_refs",
        "transfer_dst_refs",
        "transfer_input_values",
        "transfer_output_values",
        "transfer_effective_times",
    ],
)
def test_liveness_requires_complete_transfer_mappings(field):
    schedule = reduce_chain_schedule()
    inputs = resolved(CollectiveKind.REDUCE, ranks=3, slices=1)
    plan = build_buffer_plan(schedule, inputs)
    values = dict(getattr(plan, field))
    del values["reduce-a0-first"]

    with pytest.raises(SemanticError, match="every transfer requires"):
        verify_buffer_liveness(
            schedule,
            replace(plan, **{field: values}),
            inputs,
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda plan: replace(
                plan,
                transfer_src_refs={
                    **plan.transfer_src_refs,
                    "reduce-a0-first": replace(
                        plan.transfer_src_refs["reduce-a0-first"],
                        rank=1,
                    ),
                },
            ),
            "reference rank",
        ),
        (
            lambda plan: replace(
                plan,
                transfer_src_refs={
                    **plan.transfer_src_refs,
                    "reduce-a0-first": replace(
                        plan.transfer_src_refs["reduce-a0-first"],
                        valid_until=0.0,
                    ),
                },
            ),
            "outside its live interval",
        ),
        (
            lambda plan: replace(
                plan,
                transfer_dst_refs={
                    **plan.transfer_dst_refs,
                    "reduce-a0-first": replace(
                        plan.transfer_dst_refs["reduce-a0-first"],
                        valid_from=3.0,
                    ),
                },
            ),
            "becomes valid too late",
        ),
        (
            lambda plan: replace(
                plan,
                transfer_input_values={
                    **plan.transfer_input_values,
                    "reduce-a0-first": RawValue(0),
                },
            ),
            "input contributors",
        ),
        (
            lambda plan: replace(
                plan,
                transfer_accumulator_refs={
                    key: value
                    for key, value in plan.transfer_accumulator_refs.items()
                    if key != "reduce-a0-first"
                },
            ),
            "initialized accumulator",
        ),
        (
            lambda plan: replace(
                plan,
                transfer_accumulator_refs={
                    **plan.transfer_accumulator_refs,
                    "reduce-a0-first": PhysicalRef(
                        1,
                        "i",
                        0,
                        0.0,
                        float("inf"),
                    ),
                },
            ),
            "addresses differ",
        ),
        (
            lambda plan: replace(
                plan,
                transfer_accumulator_refs={
                    **plan.transfer_accumulator_refs,
                    "reduce-a0-first": replace(
                        plan.transfer_accumulator_refs["reduce-a0-first"],
                        valid_until=1.0,
                    ),
                },
            ),
            "not live before reduction",
        ),
        (
            lambda plan: replace(
                plan,
                transfer_output_values={
                    **plan.transfer_output_values,
                    "reduce-a0-first": RawValue(2),
                },
            ),
            "aggregate value",
        ),
        (
            lambda plan: replace(
                plan,
                transfer_output_values={
                    **plan.transfer_output_values,
                    "reduce-a0-first": AggregateValue(
                        0,
                        frozenset({0, 1}),
                        99,
                    ),
                },
            ),
            "add disjoint",
        ),
        (
            lambda plan: replace(
                plan,
                local_copies=(
                    replace(
                        plan.local_copies[0],
                        src_ref=replace(
                            plan.local_copies[0].src_ref,
                            valid_from=1.0,
                        ),
                    ),
                )
                + plan.local_copies[1:],
            ),
            "copy source",
        ),
        (
            lambda plan: replace(
                plan,
                local_copies=(
                    replace(
                        plan.local_copies[0],
                        dst_ref=replace(
                            plan.local_copies[0].dst_ref,
                            valid_from=1.0,
                        ),
                    ),
                )
                + plan.local_copies[1:],
            ),
            "copy destination",
        ),
    ],
)
def test_liveness_rejects_invalid_transfer_semantics(mutation, match):
    schedule = reduce_chain_schedule()
    inputs = resolved(CollectiveKind.REDUCE, ranks=3, slices=1)
    plan = build_buffer_plan(schedule, inputs)

    with pytest.raises(SemanticError, match=match):
        verify_buffer_liveness(schedule, mutation(plan), inputs)


@pytest.mark.parametrize(
    "field",
    ["final_output_refs", "final_values", "final_value_refs"],
)
def test_liveness_requires_complete_final_output_mappings(field):
    schedule = reduce_chain_schedule()
    inputs = resolved(CollectiveKind.REDUCE, ranks=3, slices=1)
    plan = build_buffer_plan(schedule, inputs)
    values = dict(getattr(plan, field))
    values.pop(next(iter(values)))

    with pytest.raises(SemanticError, match="final"):
        verify_buffer_liveness(
            schedule,
            replace(plan, **{field: values}),
            inputs,
        )


def test_liveness_rejects_out_of_place_network_write_to_input():
    schedule = reduce_chain_schedule()
    inputs = resolved(CollectiveKind.REDUCE, ranks=3, slices=1)
    plan = build_buffer_plan(schedule, inputs)
    input_ref = PhysicalRef(1, "i", 0, 0.0, float("inf"))

    with pytest.raises(SemanticError, match="input buffer is modified"):
        verify_buffer_liveness(
            schedule,
            replace(
                plan,
                transfer_dst_refs={
                    **plan.transfer_dst_refs,
                    "reduce-a0-first": input_ref,
                },
                transfer_accumulator_refs={
                    **plan.transfer_accumulator_refs,
                    "reduce-a0-first": input_ref,
                },
            ),
            inputs,
        )


def test_liveness_rejects_out_of_place_local_copy_to_input():
    schedule = reduce_chain_schedule()
    inputs = resolved(CollectiveKind.REDUCE, ranks=3, slices=1)
    plan = build_buffer_plan(schedule, inputs)

    with pytest.raises(SemanticError, match="input buffer is modified"):
        verify_buffer_liveness(
            schedule,
            replace(
                plan,
                local_copies=(
                    replace(
                        plan.local_copies[0],
                        dst_ref=PhysicalRef(
                            1,
                            "i",
                            0,
                            0.0,
                            float("inf"),
                        ),
                    ),
                )
                + plan.local_copies[1:],
            ),
            inputs,
        )
