import pytest

from vericcl.errors import InputValidationError
from tests.unit.input.v5_support import request_documents, write_request
from tests.unit.input.test_v5_contract import load


def test_default_options_leave_auxiliary_and_optional_paths_off(tmp_path):
    docs = request_documents()
    docs["atom"] = {"M_switch": 8192}
    result = load(write_request(tmp_path, docs))
    config = result.request_config
    assert config["atom"]["synthesis_method"] == "auto"
    assert result.selected_synthesis_method == "hierarchy"
    assert result.solver.max_channels == 7
    assert config["sketch"]["runtime_compatibility"]["enabled"] is True
    assert config["sketch"]["trace_and_measure"]["enabled"] is False
    assert config["sketch"]["history"]["schedule_reuse"] is False
    assert config["sketch"]["history"]["calibration_reuse"] is False
    assert "live" not in config["sketch"]
    assert config["atom"]["manual"]["enabled"] is False
    assert not result.strategies.symmetry and not result.strategies.shortest_paths
    assert not result.strategies.batching


def test_auto_method_uses_user_threshold_including_equality(tmp_path):
    docs = request_documents()
    docs["atom"] = {"M_switch": 4096}
    assert load(write_request(tmp_path, docs)).selected_synthesis_method == "constructive_trees"
    docs["atom"] = {}
    with pytest.raises(InputValidationError, match="M_switch"):
        load(write_request(tmp_path, docs))


@pytest.mark.parametrize("channels", [8, 16])
def test_large_channel_domain_is_only_explicit(tmp_path, channels):
    docs = request_documents()
    docs["sketch"]["solver"]["max_channels"] = channels
    assert load(write_request(tmp_path, docs)).solver.max_channels == channels
    docs["sketch"]["solver"]["max_channels"] = 17
    with pytest.raises(InputValidationError):
        load(write_request(tmp_path, docs))


@pytest.mark.parametrize("datatype", ["float16", "float32", "float64", "bfloat16", "int8", "uint8", "int32", "uint32", "int64", "uint64"])
def test_ten_datatypes_are_capability_requests_not_silent_substitutions(tmp_path, datatype):
    docs = request_documents()
    docs["sketch"]["collective"]["datatype"] = datatype
    result = load(write_request(tmp_path, docs))
    assert result.collective.datatype == datatype
    assert result.required_capabilities["datatype"] == datatype


@pytest.mark.parametrize("operation", ["sum", "prod", "min", "max", "avg"])
def test_five_reduction_operations_are_preserved(tmp_path, operation):
    docs = request_documents()
    docs["sketch"]["collective"]["reduction_op"] = operation
    assert load(write_request(tmp_path, docs)).collective.reduction_op == operation


@pytest.mark.parametrize("operator,field", [("allgather", "root"), ("allreduce", "root"),
    ("broadcast", "reduction_op"), ("alltoall", "reduction_op")])
def test_inapplicable_semantic_fields_must_be_unspecified(tmp_path, operator, field):
    docs = request_documents(operator)
    docs["sketch"]["collective"][field] = 0 if field == "root" else "sum"
    with pytest.raises(InputValidationError):
        load(write_request(tmp_path, docs))
    docs["sketch"]["collective"][field] = None
    assert load(write_request(tmp_path, docs))


@pytest.mark.parametrize("name", ["syccl", "forestcoll"])
def test_non_vericcl_method_names_are_not_accepted(tmp_path, name):
    docs = request_documents()
    docs["atom"]["synthesis_method"] = name
    with pytest.raises(InputValidationError):
        load(write_request(tmp_path, docs))


def test_shortest_paths_is_rejected_for_constructive_trees(tmp_path):
    docs = request_documents()
    docs["atom"].update(synthesis_method="constructive_trees", strategies={"shortest_paths": True})
    with pytest.raises(InputValidationError, match="shortest_paths"):
        load(write_request(tmp_path, docs))


def test_live_enabled_requires_migration(tmp_path):
    docs = request_documents()
    docs["sketch"]["live"] = {"enabled": True}
    with pytest.raises(InputValidationError, match="live.*removed"):
        load(write_request(tmp_path, docs))


def test_runtime_mapping_and_buffsize_are_explicit_and_valid(tmp_path):
    docs = request_documents()
    docs["runtime"]["environment"]["NCCL_BUFFSIZE"] = str(4 * 1024**3)
    with pytest.raises(InputValidationError, match="NCCL_BUFFSIZE"):
        load(write_request(tmp_path, docs))
    docs["runtime"]["environment"]["NCCL_BUFFSIZE"] = "16777216"
    docs["runtime"]["rank_map"][1]["device"] = 0
    with pytest.raises(InputValidationError, match="device"):
        load(write_request(tmp_path, docs))
