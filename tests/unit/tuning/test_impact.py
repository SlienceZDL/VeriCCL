import pytest

from vericcl.errors import SemanticError
from vericcl.tuning.impact import compute_impact_closure

from tests.unit.tuning.helpers import impact_case


pytestmark = pytest.mark.phase05


def test_impact_closure_reaches_dependency_lane_link_and_shared_resource_fixpoint():
    schedule, topology = impact_case()

    result = compute_impact_closure(
        schedule,
        frozenset({"changed"}),
        topology,
    )

    assert result.seed_transfer_ids == frozenset({"changed"})
    assert result.transfer_ids == frozenset(
        {
            "changed",
            "same-lane-later",
            "same-link",
            "shared-resource",
            "dependent",
            "recursive",
        }
    )
    assert "same_lane_successor" in result.reasons["same-lane-later"]
    assert "directed_link_concurrency" in result.reasons["same-link"]
    assert "shared_resource:nic" in result.reasons["shared-resource"]
    assert "dependency" in result.reasons["recursive"]


def test_empty_impact_seed_is_valid_and_unknown_seed_is_rejected():
    schedule, topology = impact_case()

    result = compute_impact_closure(schedule, frozenset(), topology)
    assert result.transfer_ids == frozenset()

    with pytest.raises(SemanticError, match="unknown"):
        compute_impact_closure(
            schedule,
            frozenset({"missing"}),
            topology,
        )
