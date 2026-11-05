from dataclasses import replace

import pytest

from vericcl.errors import SemanticError
from vericcl.verification.model import (
    CheckResult,
    ValidationReport,
    ValidationStatus,
)


pytestmark = pytest.mark.phase05


DIMENSIONS = (
    "input",
    "semantic",
    "state",
    "topology",
    "timing",
    "resource",
    "buffer",
    "endpoint",
    "deadlock",
    "xml",
    "bdd",
    "simulation",
    "runtime",
    "online",
)


def check(dimension, status=ValidationStatus.VALID):
    return CheckResult(
        dimension=dimension,
        status=status,
        code="{}_{}".format(dimension, status.value),
        message="{} check is {}".format(dimension, status.value),
        evidence={"dimension": dimension},
    )


def report_with(**statuses):
    values = {
        dimension: check(
            dimension,
            statuses.get(dimension, ValidationStatus.VALID),
        )
        for dimension in DIMENSIONS
    }
    return ValidationReport(**values)


def test_warning_does_not_replace_semantic_status():
    report = report_with(runtime=ValidationStatus.WARNING)

    assert report.overall_status is ValidationStatus.VALID
    assert report.semantic.status is ValidationStatus.VALID
    assert report.runtime_compatible is False
    assert report.eligible_for_selection is False


def test_online_warning_preserves_candidate_selection_eligibility():
    report = report_with(online=ValidationStatus.WARNING)

    assert report.overall_status is ValidationStatus.VALID
    assert report.eligible_for_selection is True
    assert report.online_validated is False


def test_bdd_analysis_error_blocks_selection_without_invalidating_semantics():
    report = report_with(bdd=ValidationStatus.ANALYSIS_ERROR)

    assert report.overall_status is ValidationStatus.VALID
    assert report.semantic.status is ValidationStatus.VALID
    assert report.eligible_for_selection is False


def test_fatal_input_and_invalid_correctness_have_distinct_overall_statuses():
    assert (
        report_with(input=ValidationStatus.FATAL).overall_status
        is ValidationStatus.FATAL
    )
    assert (
        report_with(state=ValidationStatus.INVALID).overall_status
        is ValidationStatus.INVALID
    )


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("dimension", "", "dimension"),
        ("status", "valid", "status"),
        ("code", "", "code"),
        ("message", "", "message"),
        ("evidence", None, "evidence"),
    ],
)
def test_check_result_rejects_invalid_fields(field, value, message):
    with pytest.raises(SemanticError, match=message):
        replace(check("semantic"), **{field: value})


def test_report_requires_matching_dimension_results():
    values = {
        dimension: check(dimension) for dimension in DIMENSIONS
    }
    values["semantic"] = check("state")

    with pytest.raises(SemanticError, match="semantic"):
        ValidationReport(**values)
