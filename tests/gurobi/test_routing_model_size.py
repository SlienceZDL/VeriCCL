from dataclasses import replace

import pytest

from vericcl.input.loader import resolve_inputs
from vericcl.input.models import ObjectiveMode
from vericcl.planner.model import PlanNode, PlanningMode, StageInterface
from vericcl.semantics.collective import CollectiveKind, CollectiveSpec, OutputSlot
from vericcl.solver.budget import ModelBudget
from vericcl.solver.demands import build_solver_problem
from vericcl.solver.routing_milp import _build_route_model, solve_route_milp
from vericcl.solver.templates import build_solver_templates
from vericcl.topology.loader import topology_from_mapping

from tests.gurobi.helpers import EXAMPLES, require_gurobi_license


pytestmark = [pytest.mark.phase03, pytest.mark.gurobi]


def _representative_case(slice_count):
    inputs = resolve_inputs(
        EXAMPLES / "topo" / "two_rank.json",
        EXAMPLES / "sketch" / "allreduce_8m_1m.json",
        EXAMPLES / "atom" / "default.json",
    )
    inputs = replace(
        inputs,
        rank_count=3,
        hyperparameters=replace(
            inputs.hyperparameters,
            total_size_bytes=slice_count * 1024,
            slice_size_bytes=1024,
        ),
    )
    topology = topology_from_mapping(
        {
            "ranks": 3,
            "nodes": [{"id": 0, "ranks": [0, 1, 2], "gateways": []}],
            "directed_links": [
                {
                    "src": src,
                    "dst": dst,
                    "alpha": 1,
                    "invbw": 2,
                    "max_channels": 4,
                }
                for src in range(3)
                for dst in range(3)
                if src != dst
            ],
            "shared_resources": [],
        }
    )
    logical_input = {
        OutputSlot(0, position): frozenset({position})
        for position in range(slice_count)
    }
    logical_output = {
        OutputSlot(rank, position): frozenset({position})
        for rank in range(3)
        for position in range(slice_count)
    }
    node = PlanNode(
        node_id="route-size",
        stage_id=0,
        local_collective=CollectiveSpec(
            kind=CollectiveKind.BROADCAST,
            datatype="float32",
            root=0,
        ),
        communication_group=(0, 1, 2),
        logical_input=StageInterface(logical_input),
        logical_output=StageInterface(logical_output),
        allowed_links=frozenset(topology.links),
        shared_resource_ids=frozenset(),
    )
    problem = build_solver_problem(node, inputs, topology)
    templates = build_solver_templates((problem,), PlanningMode.DIRECT)
    assert len(templates) == 1
    assert len(templates[0].members) == slice_count
    return inputs, topology, templates[0]


def test_route_model_size_is_exactly_slice_invariant_and_has_no_schedule_vars():
    require_gurobi_license()
    counts = []
    names = []
    for slice_count in (8, 16, 64, 128):
        inputs, topology, template = _representative_case(slice_count)
        pattern = solve_route_milp(
            template,
            inputs,
            topology,
            channel_count=4,
            objective=ObjectiveMode.LATENCY,
            budget=ModelBudget(seconds=30, started_at=0, deadline=30),
        )
        counts.append(
            (
                pattern.model_stats.variable_count,
                pattern.model_stats.constraint_count,
                pattern.model_stats.general_constraint_count,
            )
        )
        model, _, _ = _build_route_model(
            template,
            inputs,
            topology,
            channel_count=4,
            objective=ObjectiveMode.LATENCY,
            budget=ModelBudget(seconds=30, started_at=0, deadline=30),
            warm_start=None,
        )
        names.append(tuple(variable.VarName for variable in model.getVars()))
        model.dispose()

    assert counts == [counts[0]] * 4
    assert names == [names[0]] * 4
    forbidden_fragments = (
        "channel-",
        "start-",
        "end-",
        "st-time",
        "ed-time",
        "lane-order",
        "resource-order",
        "-both",
        "-first",
        "-second",
    )
    assert not any(
        fragment in name
        for name in names[0]
        for fragment in forbidden_fragments
    )
