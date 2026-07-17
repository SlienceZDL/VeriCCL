from pathlib import Path

import pytest

from vericcl.cli.online import build_online_context_factory
from vericcl.errors import InputValidationError
from vericcl.verification.online.pipeline import OnlineContext
from vericcl.verification.pipeline import validate_and_lower_candidate

from tests.unit.verification.helpers import inputs, topology
from tests.unit.xml.helpers import two_rank_allreduce_schedule


pytestmark = pytest.mark.phase07


def _environment(**changes):
    values = {
        "VERICCL_MSCCL_BUILD_DIR": "/tmp/msccl",
        "VERICCL_NCCL_TESTS_BUILD_DIR": "/tmp/nccl-tests",
        "VERICCL_CLOCK_SYNC_BINARY": "/tmp/vericcl-clock-sync",
        "VERICCL_ONLINE_INTER_NODE": "0",
        "VERICCL_MAX_CLOCK_UNCERTAINTY_US": "5.0",
    }
    values.update(changes)
    return values


def test_online_factory_requires_explicit_runtime_paths():
    with pytest.raises(InputValidationError, match="MSCCL"):
        build_online_context_factory({})


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"VERICCL_ONLINE_INTER_NODE": "yes"}, "zero or one"),
        ({"VERICCL_MAX_CLOCK_UNCERTAINTY_US": "invalid"}, "numeric"),
        ({"VERICCL_MAX_CLOCK_UNCERTAINTY_US": "-1"}, "non-negative"),
        ({"VERICCL_ONLINE_INTER_NODE": "1"}, "MPI_LAUNCHER"),
    ),
)
def test_online_factory_rejects_invalid_environment(changes, message):
    with pytest.raises(InputValidationError, match=message):
        build_online_context_factory(_environment(**changes))


def test_online_factory_builds_exact_collective_request(tmp_path):
    schedule = two_rank_allreduce_schedule()
    input_value = inputs()
    outcome = validate_and_lower_candidate(
        schedule,
        input_value,
        topology(),
    )
    factory = build_online_context_factory(_environment())

    context = factory(
        outcome.artifact,
        schedule,
        input_value,
        tmp_path / "schedule.xml",
        tmp_path / "traces",
        True,
        30.0,
    )

    assert isinstance(context, OnlineContext)
    assert context.request.kind == input_value.collective.kind
    assert context.request.message_size_bytes == (
        input_value.hyperparameters.total_size_bytes
    )
    assert context.request.datatype == "float"
    assert context.request.reduction_op == "sum"
    assert context.request.inplace is False
    assert context.xml_paths == (tmp_path / "schedule.xml",)
    assert context.trace_file_prefix == tmp_path / "traces" / "msccl-step"
    assert context.online_tuning_requested is True
    assert context.max_clock_uncertainty_us == 5.0
