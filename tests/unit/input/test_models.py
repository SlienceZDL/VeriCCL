from dataclasses import FrozenInstanceError

import pytest

from vericcl.input.models import Hyperparameters, ObjectiveMode, SolverConfig
from vericcl.semantics.collective import CollectiveKind, CollectiveSpec


pytestmark = pytest.mark.phase01


def test_hyperparameters_derive_slice_count():
    hyperparameters = Hyperparameters(
        total_size_bytes=8 * 1024 * 1024,
        slice_size_bytes=1024 * 1024,
    )

    assert hyperparameters.slice_count == 8


def test_hyperparameter_defaults_match_public_contract():
    hyperparameters = Hyperparameters(total_size_bytes=8, slice_size_bytes=1)

    assert hyperparameters.objective_mode is ObjectiveMode.AUTO
    assert hyperparameters.max_calibration_channels == 32
    assert hyperparameters.min_expected_improvement == 0.01
    assert hyperparameters.min_tuning_improvement == 0.01
    assert hyperparameters.max_tuning_iterations == 20
    assert hyperparameters.total_verification_timeout_s == 10800
    assert hyperparameters.force_recalibrate is False


def test_solver_defaults_match_public_contract():
    solver = SolverConfig()

    assert solver.total_solve_timeout_s == 10800
    assert solver.per_model_timeout_s == 1800
    assert solver.mip_gap == 1e-4
    assert solver.require_proven_optimal is False
    assert solver.solver_seed == 0
    assert solver.max_channels == 32
    assert solver.max_threads_per_model == 12
    assert solver.max_parallel_models == 4
    assert solver.force_resolve is False


def test_collective_spec_defaults_to_out_of_place():
    spec = CollectiveSpec(
        kind=CollectiveKind.ALL_REDUCE,
        datatype="float32",
        reduction_op="sum",
    )

    assert spec.inplace is False


def test_input_models_are_immutable():
    hyperparameters = Hyperparameters(total_size_bytes=8, slice_size_bytes=1)

    with pytest.raises(FrozenInstanceError):
        hyperparameters.total_size_bytes = 16
