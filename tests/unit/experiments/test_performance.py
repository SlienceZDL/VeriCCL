from dataclasses import replace

import pytest

from vericcl.errors import SemanticError
from vericcl.experiments.performance import (
    ActivationEvidence,
    evaluate_msccl_activation,
)
from vericcl.verification.online.model import NcclTestMeasurement, NcclTestRun


def _run(out_busbw, in_busbw):
    return NcclTestRun(
        message_size_bytes=4 * 1024 * 1024,
        element_count=1024 * 1024,
        datatype="float",
        metadata_fields=("none", "-1"),
        out_of_place=NcclTestMeasurement(10.0, 60.0, out_busbw, 0),
        in_place=NcclTestMeasurement(11.0, 65.0, in_busbw, 0),
    )


def test_activation_requires_info_and_five_percent_busbw_difference():
    run = _run(out_busbw=70.0, in_busbw=75.0)

    evidence = evaluate_msccl_activation(
        "NCCL INFO Connected 1 MSCCL algorithms\n",
        run,
    )

    assert evidence.info_loaded is True
    assert evidence.relative_busbw_difference == pytest.approx(5.0 / 75.0)
    assert evidence.confirmed is True


def test_activation_is_unconfirmed_without_info():
    evidence = evaluate_msccl_activation("", _run(70.0, 75.0))

    assert evidence.info_loaded is False
    assert evidence.confirmed is False


def test_activation_requires_both_placements_and_threshold():
    out_only = replace(_run(70.0, 75.0), in_place=None)

    assert evaluate_msccl_activation(
        "NCCL INFO Connected 1 MSCCL algorithms\n",
        out_only,
    ).confirmed is False
    assert evaluate_msccl_activation(
        "NCCL INFO Connected 1 MSCCL algorithms\n",
        _run(70.0, 73.0),
    ).confirmed is False


def test_activation_models_reject_invalid_boundaries():
    evidence = ActivationEvidence(True, 0.1, 0.05, True)

    for changes in (
        {"info_loaded": "yes"},
        {"relative_busbw_difference": -1.0},
        {"threshold": 2.0},
        {"confirmed": "yes"},
    ):
        with pytest.raises(SemanticError):
            replace(evidence, **changes)
    with pytest.raises(SemanticError):
        evaluate_msccl_activation(object(), _run(70.0, 75.0))
    with pytest.raises(SemanticError):
        evaluate_msccl_activation("", object())
