from dataclasses import replace

import pytest

from vericcl.errors import SemanticError
from vericcl.semantics.collective import CollectiveKind
from vericcl.xml.buffers import build_buffer_plan
from vericcl.xml.dependencies import build_transfer_dag
from vericcl.xml.endpoints import lower_endpoints
from vericcl.xml.model import RawValue

from tests.unit.xml.helpers import (
    allreduce_star_schedule,
    inplace_alltoall_overwrite_schedule,
    resolved,
    send_relay_schedule,
)


pytestmark = pytest.mark.phase04


def test_relay_receive_precedes_a_distinct_send_node():
    schedule = send_relay_schedule()
    buffers = build_buffer_plan(
        schedule,
        resolved(CollectiveKind.BROADCAST, ranks=3, slices=1),
    )
    program = lower_endpoints(schedule, buffers)

    dag = build_transfer_dag(program, schedule, buffers)

    assert "relay-first" in dag.predecessors["relay-second"]
    assert dag.nodes["relay-first"].endpoint_ids != dag.nodes[
        "relay-second"
    ].endpoint_ids
    assert "path" in dag.edge_reasons[("relay-first", "relay-second")]


def test_three_reduce_contributors_remain_direct_semantic_predecessors():
    schedule = allreduce_star_schedule()
    buffers = build_buffer_plan(
        schedule,
        resolved(CollectiveKind.ALL_REDUCE, ranks=4, slices=1),
    )
    program = lower_endpoints(schedule, buffers)

    dag = build_transfer_dag(program, schedule, buffers)

    expected = {"reduce-star-1", "reduce-star-2", "reduce-star-3"}
    for destination in range(1, 4):
        consumer = "allreduce-send-{}".format(destination)
        assert expected <= dag.predecessors[consumer]
        assert all(
            "semantic" in dag.edge_reasons[(predecessor, consumer)]
            for predecessor in expected
        )


def test_shared_rrc_accumulator_versions_form_a_buffer_chain():
    schedule = allreduce_star_schedule()
    buffers = build_buffer_plan(
        schedule,
        resolved(CollectiveKind.ALL_REDUCE, ranks=4, slices=1),
    )
    program = lower_endpoints(schedule, buffers)

    dag = build_transfer_dag(program, schedule, buffers)

    assert "reduce-star-1" in dag.predecessors["reduce-star-2"]
    assert "reduce-star-2" in dag.predecessors["reduce-star-3"]
    assert "buffer_state" in dag.edge_reasons[
        ("reduce-star-2", "reduce-star-3")
    ]


def test_copy_initialization_precedes_rrc_use():
    schedule = allreduce_star_schedule()
    buffers = build_buffer_plan(
        schedule,
        resolved(CollectiveKind.ALL_REDUCE, ranks=4, slices=1),
    )
    program = lower_endpoints(schedule, buffers)

    dag = build_transfer_dag(program, schedule, buffers)

    initializing_copy = next(
        copy.copy_id
        for copy in buffers.local_copies
        if copy.reason == "initialize reduction accumulator"
    )
    assert initializing_copy in dag.predecessors["reduce-star-1"]
    assert "buffer_init" in dag.edge_reasons[
        (initializing_copy, "reduce-star-1")
    ]


def test_dependency_builder_rejects_a_cycle():
    schedule = send_relay_schedule()
    buffers = build_buffer_plan(
        schedule,
        resolved(CollectiveKind.BROADCAST, ranks=3, slices=1),
    )
    program = lower_endpoints(schedule, buffers)
    cyclic = replace(
        schedule,
        metadata={
            **schedule.metadata,
            "semantic_predecessors": {
                "relay-first": ("relay-second",),
                "relay-second": ("relay-first",),
            },
        },
    )

    with pytest.raises(SemanticError, match="cycle"):
        build_transfer_dag(program, cyclic, buffers)


def test_dependency_builder_rejects_crossed_accumulator_state():
    schedule = allreduce_star_schedule()
    buffers = build_buffer_plan(
        schedule,
        resolved(CollectiveKind.ALL_REDUCE, ranks=4, slices=1),
    )
    program = lower_endpoints(schedule, buffers)
    accumulator_values = dict(buffers.transfer_accumulator_values)
    accumulator_values["reduce-star-2"] = RawValue(0)

    with pytest.raises(SemanticError, match="accumulator state"):
        build_transfer_dag(
            program,
            schedule,
            replace(
                buffers,
                transfer_accumulator_values=accumulator_values,
            ),
        )


def test_inplace_preservation_is_an_antidependency_of_receive_overwrite():
    schedule = inplace_alltoall_overwrite_schedule()
    buffers = build_buffer_plan(
        schedule,
        resolved(CollectiveKind.ALL_TO_ALL, inplace=True),
    )
    program = lower_endpoints(schedule, buffers)

    dag = build_transfer_dag(program, schedule, buffers)

    for copy in buffers.local_copies:
        if copy.reason != "preserve live in-place input":
            continue
        overwritten_by = "incoming" if copy.rank == 1 else "outgoing"
        assert copy.copy_id in dag.predecessors[overwritten_by]
        assert "buffer_antidependency" in dag.edge_reasons[
            (copy.copy_id, overwritten_by)
        ]


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda dag: replace(
                dag,
                predecessors={"relay-first": frozenset()},
            ),
            "cover every node",
        ),
        (
            lambda dag: replace(
                dag,
                predecessors={
                    **dag.predecessors,
                    "relay-first": frozenset({"relay-first"}),
                },
            ),
            "invalid edge",
        ),
        (
            lambda dag: replace(
                dag,
                edge_reasons={
                    ("relay-second", "relay-first"): frozenset({"test"})
                },
            ),
            "does not match",
        ),
        (
            lambda dag: replace(dag, topological_order=("relay-first",)),
            "incomplete",
        ),
        (
            lambda dag: replace(
                dag,
                topological_order=tuple(reversed(dag.topological_order)),
            ),
            "cycle",
        ),
    ],
)
def test_transfer_dag_rejects_invalid_graph_records(mutation, match):
    schedule = send_relay_schedule()
    buffers = build_buffer_plan(
        schedule,
        resolved(CollectiveKind.BROADCAST, ranks=3, slices=1),
    )
    dag = build_transfer_dag(
        lower_endpoints(schedule, buffers),
        schedule,
        buffers,
    )

    with pytest.raises(SemanticError, match=match):
        mutation(dag)
