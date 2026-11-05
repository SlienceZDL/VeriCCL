from dataclasses import replace

import pytest

from vericcl.solver.budget import ModelBudget
from tests.scip.helpers import fresh_budget
from vericcl.solver.scip_api import ScipAdapter
from vericcl.tuning.impact import compute_impact_closure
from vericcl.tuning.local_milp import solve_local_repair
from vericcl.tuning.model import RepairStatus

from tests.unit.tuning.helpers import overlay, waiting_case


pytestmark = [pytest.mark.phase05, pytest.mark.scip]


def test_local_repair_component_fixture_requires_full_collective_contract():
    schedule, topology, inputs, hint = waiting_case()
    impact = compute_impact_closure(
        schedule,
        frozenset({hint.waiting_transfer_id}),
        topology,
    )

    result = solve_local_repair(
        schedule,
        hint,
        impact,
        replace(
            overlay(),
            milp_parameters=(("MIPGap", 0.1),),
        ),
        topology,
        inputs,
        fresh_budget(10.0),
    )

    if not ScipAdapter.available():
        assert result.status is RepairStatus.NOT_RUN
    else:
        assert result.status is RepairStatus.NOT_RUN
        assert result.schedule is None
        assert result.evidence["code"] == "local_repair_incomplete_collective_contract"
    assert result.evidence["scope"] == "local"


def test_local_repair_model_excludes_illegal_shorter_candidate():
    schedule, topology, inputs, hint = waiting_case()
    legal_id = hint.candidate_flow_ids[0]
    illegal_id = "illegal-shorter"
    candidates = replace(
        hint,
        candidate_flow_ids=(illegal_id, legal_id),
        candidate_paths={
            illegal_id: (1, 3),
            legal_id: hint.candidate_paths[legal_id],
        },
        candidate_first_lanes={
            illegal_id: hint.bottleneck_lane,
            legal_id: hint.candidate_first_lanes[legal_id],
        },
    )
    impact = compute_impact_closure(
        schedule,
        frozenset({hint.waiting_transfer_id}),
        topology,
    )

    result = solve_local_repair(
        schedule,
        candidates,
        impact,
        overlay(),
        topology,
        inputs,
        fresh_budget(10.0),
    )

    if not ScipAdapter.available():
        assert result.status is RepairStatus.NOT_RUN
    else:
        assert result.status is RepairStatus.NOT_RUN
        assert result.evidence["selected_candidate_flow_id"] == legal_id
        assert result.evidence["legal_candidate_count"] == 1
