import pytest
from hypothesis import given, strategies as st

from vericcl.errors import SemanticError
from vericcl.solver import demands as demand_module
from vericcl.solver.demands import TransferDemand
from vericcl.topology.model import LinkKey


pytestmark = pytest.mark.phase03


BASE_ID = "identity-node-a00000000-r00000000-l00000001"


def _demand(contributors, members):
    link = LinkKey(0, 1)
    return TransferDemand(
        demand_id=BASE_ID,
        node_id="identity-node",
        stage_id=0,
        root_rank=0,
        required_leaf_rank=1,
        logical_position=0,
        contributors=frozenset(contributors),
        member_slice_ids=frozenset(members),
        allowed_links=frozenset({link}),
        legal_links=frozenset({link}),
        forbidden_members=(),
        candidate_paths=((0, 1),),
        reduction_dual=False,
    )


def _identity_map(demands):
    return {
        (
            tuple(sorted(demand.contributors)),
            tuple(sorted(demand.member_slice_ids)),
        ): demand.demand_id
        for demand in demands
    }


@given(st.permutations((0, 1, 2)))
def test_collision_ordinals_follow_canonical_identity_not_input_order(order):
    demands = (
        _demand({0}, {0}),
        _demand({0, 8}, {0, 8}),
        _demand({0, 8}, {8}),
    )

    assigned = demand_module._assign_demand_ids(
        tuple(demands[index] for index in order)
    )

    assert _identity_map(assigned) == {
        ((0,), (0,)): "{}-v00000000".format(BASE_ID),
        ((0, 8), (0, 8)): "{}-v00000001".format(BASE_ID),
        ((0, 8), (8,)): "{}-v00000002".format(BASE_ID),
    }
    assert len({demand.demand_id for demand in assigned}) == 3


def test_singletons_keep_legacy_ids_and_collision_ids_have_constant_length():
    singleton = demand_module._assign_demand_ids((_demand({0}, {0}),))
    large_contributors = frozenset(index * 8 for index in range(1024))
    collisions = demand_module._assign_demand_ids(
        (
            _demand({0}, {0}),
            _demand(large_contributors, large_contributors),
        )
    )

    assert singleton[0].demand_id == BASE_ID
    assert {len(demand.demand_id) for demand in collisions} == {
        len(BASE_ID) + len("-v00000000")
    }


def test_duplicate_canonical_identity_is_rejected_instead_of_renumbered():
    demand = _demand({0}, {0})

    with pytest.raises(SemanticError, match="canonical demand identity"):
        demand_module._assign_demand_ids((demand, demand))
