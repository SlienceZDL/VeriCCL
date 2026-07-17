from dataclasses import replace
import subprocess
import sys

import pytest

from vericcl.errors import SemanticError
from vericcl.input.models import ForbiddenTransfer
from vericcl.tuning.model import TuningOverlay

from tests.unit.tuning.helpers import overlay, waiting_case


pytestmark = pytest.mark.phase05


def test_tuning_repair_interfaces_are_public_exports():
    from vericcl.tuning import (
        ImpactClosure,
        RepairResult,
        RepairStatus,
        TuningOverlay,
        compute_impact_closure,
        repair_flow_suffix,
        solve_local_repair,
    )

    assert all(
        value is not None
        for value in (
            ImpactClosure,
            RepairResult,
            RepairStatus,
            TuningOverlay,
            compute_impact_closure,
            repair_flow_suffix,
            solve_local_repair,
        )
    )


def test_verification_first_import_order_has_no_cycle():
    result = subprocess.run(
        (
            sys.executable,
            "-c",
            (
                "from vericcl.verification.flow_index import build_flow_index; "
                "from vericcl.tuning import repair_flow_suffix; "
                "assert build_flow_index and repair_flow_suffix"
            ),
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

def test_overlay_validation_preserves_all_parent_inputs():
    schedule, topology, inputs, _ = waiting_case()
    value = overlay()
    input_snapshot = repr(inputs)
    schedule_snapshot = repr(schedule)
    topology_snapshot = topology.isomorphism_signature

    value.validate_against(inputs, schedule, topology)

    assert repr(inputs) == input_snapshot
    assert repr(schedule) == schedule_snapshot
    assert topology.isomorphism_signature == topology_snapshot


def test_overlay_cannot_override_immutable_problem_fields():
    with pytest.raises(TypeError):
        TuningOverlay(
            overlay_id="invalid",
            parent_candidate_id=None,
            slice_size_bytes=2048,
        )
    with pytest.raises(TypeError):
        TuningOverlay(
            overlay_id="invalid",
            parent_candidate_id=None,
            collective="allreduce",
        )
    with pytest.raises(TypeError):
        TuningOverlay(
            overlay_id="invalid",
            parent_candidate_id=None,
            topology_links=(),
        )


def test_overlay_rejects_manual_hierarchy_and_user_boundary_changes():
    schedule, topology, inputs, _ = waiting_case()
    value = replace(overlay(), hierarchy_template="other")
    with pytest.raises(SemanticError, match="hierarchy"):
        value.validate_against(inputs, schedule, topology)

    too_many = replace(overlay(), channel_count=inputs.solver.max_channels + 1)
    with pytest.raises(SemanticError, match="channel"):
        too_many.validate_against(inputs, schedule, topology)

    invalid_forbidden = replace(
        overlay(),
        temporary_forbidden=frozenset(
            {ForbiddenTransfer(999, 0, 1, 0)}
        ),
    )
    with pytest.raises(SemanticError, match="forbidden"):
        invalid_forbidden.validate_against(inputs, schedule, topology)

    negative_slice = replace(
        overlay(),
        temporary_forbidden=frozenset(
            {ForbiddenTransfer(-1, 0, 1, 0)}
        ),
    )
    with pytest.raises(SemanticError, match="forbidden"):
        negative_slice.validate_against(inputs, schedule, topology)

    invalid_stage = replace(
        overlay(),
        temporary_forbidden=frozenset(
            {ForbiddenTransfer(0, 0, 1, 999)}
        ),
    )
    with pytest.raises(SemanticError, match="forbidden"):
        invalid_stage.validate_against(inputs, schedule, topology)


def test_overlay_rejects_mismatched_schedule_slice_contract():
    schedule, topology, inputs, _ = waiting_case()
    changed = replace(
        inputs,
        hyperparameters=replace(
            inputs.hyperparameters,
            total_size_bytes=2048,
        ),
    )

    with pytest.raises(SemanticError, match="slice"):
        overlay().validate_against(changed, schedule, topology)
