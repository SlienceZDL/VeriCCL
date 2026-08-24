from dataclasses import FrozenInstanceError, replace

import pytest

import vericcl.solver.routing_milp as routing_milp
from vericcl.errors import SemanticError
from vericcl.input.json_codec import canonical_json
from vericcl.input.models import ForbiddenTransfer, ObjectiveMode
from vericcl.solver.budget import ModelBudget
from vericcl.solver.lower_bounds import representative_edge_loads
from vericcl.solver.routing import RoutePattern, RoutingModelStats
from vericcl.solver.routing_milp import (
    _validate_extracted_route,
    solve_route_milp,
)
from vericcl.solver.templates import SolverTemplate, split_routing_units
from vericcl.topology.model import LinkKey

from tests.unit.solver.test_templates import (
    _inputs,
    _public_template_fixture,
    _topology,
    _tree_problem,
)


pytestmark = pytest.mark.phase03


def _template():
    unit, member = _public_template_fixture()
    return SolverTemplate(
        template_id="routing-template",
        representative=unit,
        members=(member,),
        exact_signature="routing-signature",
    )


def test_route_pattern_is_frozen_and_canonical_json_serializable():
    stats = RoutingModelStats(
        variable_count=7,
        constraint_count=11,
        general_constraint_count=0,
        build_time_s=0.125,
        optimize_time_s=0.25,
    )
    pattern = RoutePattern(
        template_id="routing-template",
        channel_count=4,
        objective_mode=ObjectiveMode.LATENCY,
        selected_edges=(LinkKey(0, 1), LinkKey(1, 2)),
        parent_edges=((0, 1), (1, 2)),
        model_stats=stats,
    )

    assert canonical_json(pattern) == (
        '{"channel_count":4,"model_stats":{"build_time_s":0.125,'
        '"constraint_count":11,"general_constraint_count":0,'
        '"optimize_time_s":0.25,"variable_count":7},'
        '"objective_mode":"latency","parent_edges":[[0,1],[1,2]],'
        '"selected_edges":[{"dst_rank":1,"src_rank":0},'
        '{"dst_rank":2,"src_rank":1}],"template_id":"routing-template"}'
    )
    with pytest.raises(FrozenInstanceError):
        pattern.channel_count = 2


def test_route_pattern_canonicalizes_edge_order():
    pattern = RoutePattern(
        template_id="routing-template",
        channel_count=4,
        objective_mode=ObjectiveMode.LATENCY,
        selected_edges=(LinkKey(1, 2), LinkKey(0, 1)),
        parent_edges=((1, 2), (0, 1)),
        model_stats=RoutingModelStats(7, 11, 0, 0.125, 0.25),
    )

    assert pattern.selected_edges == (LinkKey(0, 1), LinkKey(1, 2))
    assert pattern.parent_edges == ((0, 1), (1, 2))


def _validation_unit():
    inputs = _inputs(6, 1)
    topology = _topology(((0, 1, 2, 3, 4, 5),))
    problem = _tree_problem(inputs, topology, (0, 1, 2, 3, 4, 5), 0, 0)
    problem = replace(
        problem,
        demands=tuple(
            demand
            for demand in problem.demands
            if demand.required_leaf_rank <= 3
        ),
        infeasible_demand_ids=(),
    )
    return split_routing_units(problem)[0]


def _chain_flows(unit):
    return {
        demand.demand_id: {
            LinkKey(source, source + 1)
            for source in range(demand.required_leaf_rank)
        }
        for demand in unit.demands
    }


def test_extracted_route_validation_accepts_one_connected_legal_tree():
    unit = _validation_unit()
    flows = _chain_flows(unit)

    _validate_extracted_route(
        unit,
        {edge for demand_flow in flows.values() for edge in demand_flow},
        flows,
    )


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("unreachable", "continuous path"),
        ("multiple_parents", "multiple parents"),
        ("cycle", "enters its root"),
        ("disconnected", "unused or disconnected"),
        ("illegal", "forbidden or illegal"),
        ("forbidden", "forbidden physical transfer"),
    ),
)
def test_extracted_route_validation_rejects_invalid_solver_output(case, message):
    unit = _validation_unit()
    flows = _chain_flows(unit)
    selected = {edge for demand_flow in flows.values() for edge in demand_flow}
    leaf_one = next(
        demand for demand in unit.demands if demand.required_leaf_rank == 1
    )
    if case == "unreachable":
        flows[leaf_one.demand_id] = set()
        selected = {edge for demand_flow in flows.values() for edge in demand_flow}
    elif case == "multiple_parents":
        selected.add(LinkKey(0, 2))
    elif case == "cycle":
        selected.add(LinkKey(1, 0))
    elif case == "disconnected":
        selected.add(LinkKey(4, 5))
    elif case == "illegal":
        flows[leaf_one.demand_id].add(LinkKey(6, 7))
    else:
        forbidden = ForbiddenTransfer(
            slice_id=0,
            src_rank=0,
            dst_rank=1,
            stage_id=0,
        )
        modified = replace(
            leaf_one,
            forbidden_members=(forbidden,),
        )
        unit = replace(
            unit,
            demands=tuple(
                modified if demand is leaf_one else demand
                for demand in unit.demands
            ),
        )

    with pytest.raises(SemanticError, match=message):
        _validate_extracted_route(unit, selected, flows)


def test_representative_edge_loads_count_flows_and_normalize_by_fixed_k():
    assert representative_edge_loads(
        (LinkKey(0, 1), LinkKey(0, 1), LinkKey(1, 2)),
        channel_count=4,
    ) == ((LinkKey(0, 1), 0.5), (LinkKey(1, 2), 0.25))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("template", object()),
        ("channel_count", 0),
        ("objective_mode", "latency"),
        ("objective_mode", ObjectiveMode.AUTO),
        ("budget", object()),
    ),
)
def test_solve_route_milp_rejects_invalid_api_arguments_before_gurobi(
    field,
    value,
):
    arguments = {
        "template": _template(),
        "channel_count": 1,
        "objective_mode": ObjectiveMode.LATENCY,
        "budget": ModelBudget(seconds=1, started_at=0, deadline=1),
    }
    arguments[field] = value

    with pytest.raises(SemanticError):
        solve_route_milp(**arguments)


def test_solve_route_milp_disposes_model_when_optimization_fails(monkeypatch):
    class FailingModel:
        disposed = False

        def optimize(self):
            raise RuntimeError("optimization failed")

        def dispose(self):
            self.disposed = True

    model = FailingModel()
    monkeypatch.setattr(
        routing_milp.GurobiAdapter,
        "require",
        staticmethod(lambda: object()),
    )
    monkeypatch.setattr(
        routing_milp.GurobiAdapter,
        "model_counts",
        staticmethod(lambda current: (0, 0, 0)),
    )
    monkeypatch.setattr(
        routing_milp,
        "_build_route_model",
        lambda *args: (model, object()),
    )

    with pytest.raises(RuntimeError, match="optimization failed"):
        solve_route_milp(
            _template(),
            channel_count=1,
            objective_mode=ObjectiveMode.LATENCY,
            budget=ModelBudget(seconds=1, started_at=0, deadline=1),
        )

    assert model.disposed
