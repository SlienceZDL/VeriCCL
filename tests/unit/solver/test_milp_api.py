import pytest

from vericcl.errors import SemanticError
from vericcl.input.models import ObjectiveMode
from vericcl.solver.budget import ModelBudget
from vericcl.solver.milp import (
    _OperationValue,
    _build_schedule,
    _trees,
    solve_milp,
)
from vericcl.topology.model import LinkKey

from tests.gurobi.helpers import broadcast_problem, multihop_problem


pytestmark = pytest.mark.phase03


def _arguments():
    return {
        "problem": broadcast_problem(logical_positions=(0,)),
        "channel_count": 1,
        "objective": ObjectiveMode.LATENCY,
        "budget": ModelBudget(seconds=1, started_at=0, deadline=1),
        "warm_start": None,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("problem", object()),
        ("channel_count", 0),
        ("channel_count", 33),
        ("objective", "latency"),
        ("budget", object()),
        ("warm_start", object()),
    ],
)
def test_solve_milp_rejects_invalid_api_arguments(field, value):
    arguments = _arguments()
    arguments[field] = value

    with pytest.raises(SemanticError):
        solve_milp(**arguments)


def test_milp_schedule_materializer_preserves_existing_semantics():
    problem = multihop_problem(shared_resource=True)
    tree = _trees(problem)[0]
    links = (LinkKey(0, 1), LinkKey(1, 2))
    operations = tuple(
        _OperationValue(
            key=(tree.index, link),
            tree=tree,
            link=link,
            channel=0,
            start_time=float(index * 3),
            end_time=float((index + 1) * 3),
            duration=3.0,
            resource_slots={"shared-path": 0},
        )
        for index, link in enumerate(links)
    )
    flows = {tree.demands[0].demand_id: frozenset(links)}

    schedule = _build_schedule(
        problem,
        (tree,),
        operations,
        flows,
        channel_count=1,
    )

    assert schedule.schedule_id == "milp-multihop-milp-k01"
    assert tuple(
        transfer.transfer_id for transfer in schedule.transfers
    ) == (
        "milp-multihop-milp-t00000000",
        "milp-multihop-milp-t00000001",
    )
    first, second = schedule.transfers
    assert second.predecessor_ids == frozenset({first.transfer_id})
    assert [
        (symbol.src_rank, symbol.dst_rank)
        for symbol in second.atoms[0].path[0].symbols
    ] == [(0, 1), (1, 2)]
    assert schedule.final_state_ids == (
        "milp-multihop-r00000000-o00000000",
        "milp-multihop-r00000002-o00000000",
    )
    assert schedule.metadata["selected_flows"] == {
        tree.demands[0].demand_id: ((0, 1), (1, 2))
    }
    assert schedule.metadata["resource_slots"] == {
        first.transfer_id: {"shared-path": 0},
        second.transfer_id: {"shared-path": 0},
    }
