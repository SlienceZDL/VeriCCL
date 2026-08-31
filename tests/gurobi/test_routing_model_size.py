import re
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


_ROUTE_VARIABLE_TYPES = (
    (re.compile(r"tree-edge-e\d{4}-r\d{4}-r\d{4}"), "B"),
    (re.compile(r"flow-path-d\d{4}-p\d{4}"), "B"),
    (re.compile(r"flow-edge-d\d{4}-e\d{4}"), "B"),
    (re.compile(r"level-r\d{4}"), "C"),
    (re.compile(r"route-completion-us"), "C"),
    (re.compile(r"maximum-resource-load-us"), "C"),
)


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
    variable_shapes = []
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
        variables = tuple(
            (variable.VarName, variable.VType)
            for variable in model.getVars()
        )
        variable_shapes.append(variables)
        for name, variable_type in variables:
            matches = [
                expected_type
                for pattern, expected_type in _ROUTE_VARIABLE_TYPES
                if pattern.fullmatch(name)
            ]
            assert matches, "unexpected route variable family: {}".format(
                name
            )
            assert matches == [variable_type]
        assert len(model.getConstrs()) == model.NumConstrs
        assert all(
            constraint.Sense in {"<", "=", ">"}
            for constraint in model.getConstrs()
        )
        assert model.NumQConstrs == 0
        assert model.getQConstrs() == []
        assert model.NumSOS == 0
        assert model.getSOSs() == []
        assert model.NumGenConstrs == 0
        assert model.getGenConstrs() == []
        model.dispose()

    assert counts == [counts[0]] * 4
    assert variable_shapes == [variable_shapes[0]] * 4
