from dataclasses import replace

import pytest

from vericcl.solver.budget import ModelBudget
from vericcl.solver.gurobi_api import GurobiAdapter
from vericcl.tuning.impact import compute_impact_closure
from vericcl.tuning.local_milp import solve_local_repair
from vericcl.tuning.model import RepairStatus

from tests.unit.tuning.helpers import overlay, waiting_case


pytestmark = [pytest.mark.phase05, pytest.mark.gurobi]


def test_local_repair_model_never_expands_to_global_solve():
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
        ModelBudget(10.0, 0.0, 10.0),
    )

    if not GurobiAdapter.available():
        assert result.status is RepairStatus.NOT_RUN
    else:
        assert result.status is RepairStatus.SUCCESS
        assert result.evidence["modeled_transfer_ids"] == tuple(
            sorted(impact.transfer_ids)
        )
        assert result.evidence["fixed_transfer_count"] == (
            len(schedule.transfers) - len(impact.transfer_ids)
        )
        assert result.evidence["timing_variable_count"] == (
            2 * len(impact.transfer_ids)
        )
        assert result.evidence["model_variable_count"] == (
            len(hint.candidate_flow_ids) + 2 * len(impact.transfer_ids)
        )
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
        ModelBudget(10.0, 0.0, 10.0),
    )

    if not GurobiAdapter.available():
        assert result.status is RepairStatus.NOT_RUN
    else:
        assert result.status is RepairStatus.SUCCESS
        assert result.selected_candidate_flow_id == legal_id
        assert result.evidence["legal_candidate_count"] == 1
