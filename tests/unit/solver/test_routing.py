from dataclasses import FrozenInstanceError

import pytest

from vericcl.errors import SemanticError
from vericcl.input.json_codec import canonical_json
from vericcl.input.models import ObjectiveMode
from vericcl.solver.budget import ModelBudget
from vericcl.solver.lower_bounds import representative_edge_loads
from vericcl.solver.routing import RoutePattern, RoutingModelStats
from vericcl.solver.routing_milp import solve_route_milp
from vericcl.solver.templates import SolverTemplate
from vericcl.topology.model import LinkKey

from tests.unit.solver.test_templates import _public_template_fixture


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
