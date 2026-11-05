import json

import pytest

from vericcl.errors import InputValidationError
from tests.unit.input.v5_support import request_documents, write_request


def load(files, overrides=None, **kwargs):
    from vericcl.input.loader import load_request
    return load_request(files, overrides=overrides, **kwargs)


@pytest.mark.parametrize("operator,output,test_size", [
    ("broadcast", 4096, 4096), ("reduce", 4096, 4096),
    ("allgather", 16384, 16384), ("allreduce", 4096, 4096),
    ("alltoall", 4096, 4096), ("reduce_scatter", 1024, 4096),
])
def test_six_operators_have_explicit_input_output_and_test_size(tmp_path, operator, output, test_size):
    result = load(write_request(tmp_path, request_documents(operator)))
    assert result.hyperparameters.slice_count == 4
    assert result.size_accounting["input_bytes_per_rank"] == 4096
    assert result.size_accounting["valid_output_bytes"] == output
    assert result.size_accounting["nccl_tests_size_bytes"] == test_size
    if operator == "reduce":
        assert result.size_accounting["output_ranks"] == (0,)


def test_missing_and_null_defaults_agree_and_sources_are_retained(tmp_path):
    docs = request_documents()
    docs["sketch"]["collective"].update(datatype=None, reduction_op=None, inplace=None)
    result = load(write_request(tmp_path, docs))
    assert result.collective.datatype == "float32"
    assert result.collective.reduction_op == "sum"
    assert result.collective.inplace is False
    assert result.value_sources["sketch.collective.datatype"]["kind"] == "default"
    assert result.value_sources["sketch.hyperparameters.total_size_bytes"]["kind"] == "file"


def test_cli_fills_unspecified_values_but_concrete_conflicts_need_override(tmp_path):
    docs = request_documents()
    docs["sketch"]["collective"]["datatype"] = None
    files = write_request(tmp_path, docs)
    result = load(files, {"sketch.collective.datatype": "float64"})
    assert result.collective.datatype == "float64"
    with pytest.raises(InputValidationError, match="override-input"):
        load(files, {"sketch.hyperparameters.total_size_bytes": 8192})
    result = load(files, {"sketch.hyperparameters.total_size_bytes": 8192}, override_input=True)
    assert result.hyperparameters.slice_count == 8
    source = result.value_sources["sketch.hyperparameters.total_size_bytes"]
    assert source["kind"] == "cli" and source["previous"] == 4096
    assert json.loads(files["sketch"].read_text())["hyperparameters"]["total_size_bytes"] == 4096


@pytest.mark.parametrize("section,key", [("hyperparameters", "input_chunkup"),
    ("hyperparameters", "N"), ("hyperparameters", "force_recalibrate"), ("solver", "force_resolve")])
def test_removed_fields_are_errors_even_if_they_match_derived_values(tmp_path, section, key):
    docs = request_documents()
    docs["sketch"][section][key] = 4
    with pytest.raises(InputValidationError, match="removed"):
        load(write_request(tmp_path, docs))


@pytest.mark.parametrize("old", ["hierarchy", "constructive_trees", "milp"])
def test_removed_algorithm_booleans_are_not_guessed(tmp_path, old):
    docs = request_documents()
    docs["atom"]["strategies"] = {old: True}
    with pytest.raises(InputValidationError, match="synthesis_method"):
        load(write_request(tmp_path, docs))


def test_listed_legacy_locations_migrate_but_duplicates_are_rejected(tmp_path):
    docs = request_documents()
    docs["sketch"]["hyperparameters"]["per_model_timeout_s"] = docs["sketch"]["solver"].pop("per_model_timeout_s")
    docs["atom"]["symmetry"] = True
    files = write_request(tmp_path, docs)
    result = load(files)
    assert result.solver.per_model_timeout_s == 2
    assert result.strategies.symmetry
    assert result.value_sources["sketch.solver.per_model_timeout_s"]["original_path"] == "sketch.hyperparameters.per_model_timeout_s"
    docs["sketch"]["solver"]["per_model_timeout_s"] = 2
    with pytest.raises(InputValidationError, match="duplicate"):
        load(write_request(tmp_path, docs))


def test_duplicate_json_keys_and_unknown_nested_fields_are_rejected(tmp_path):
    files = write_request(tmp_path)
    files["atom"].write_text('{"stage_num": 1, "stage_num": 1}')
    with pytest.raises(InputValidationError, match="duplicate"):
        load(files)
    files["atom"].write_text('{"strategies": {"unknown": true}}')
    with pytest.raises(InputValidationError, match="unknown"):
        load(files)


@pytest.mark.parametrize("operator,size,slice_size", [("allreduce", 4097, 1024),
    ("allreduce", 4096, 1023), ("alltoall", 3072, 1024), ("reduce_scatter", 3072, 1024)])
def test_invalid_element_or_rank_partition_is_rejected_without_padding(tmp_path, operator, size, slice_size):
    docs = request_documents(operator)
    docs["sketch"]["hyperparameters"].update(total_size_bytes=size, slice_size_bytes=slice_size)
    with pytest.raises(InputValidationError):
        load(write_request(tmp_path, docs))


def test_auto_slice_is_pending_capacity_selection_and_constraints_need_reference(tmp_path):
    docs = request_documents()
    docs["sketch"]["hyperparameters"]["slice_size_bytes"] = None
    result = load(write_request(tmp_path, docs))
    assert result.hyperparameters.slice_size_bytes is None
    assert result.hyperparameters.slice_count is None
    docs["atom"]["forbidden_transfers"] = [[0, 0, 1, 0]]
    with pytest.raises(InputValidationError, match="reference_slice_size_bytes"):
        load(write_request(tmp_path, docs))
    docs["sketch"]["hyperparameters"]["reference_slice_size_bytes"] = 1024
    result = load(write_request(tmp_path, docs))
    assert result.hyperparameters.reference_slice_size_bytes == 1024


def test_required_enabled_module_budgets_are_not_invented(tmp_path):
    docs = request_documents()
    docs["sketch"]["calibration"] = {"enabled": True}
    with pytest.raises(InputValidationError, match="max_online_validation"):
        load(write_request(tmp_path, docs))


def test_legacy_total_solver_budget_moves_to_initial_synthesis_only(tmp_path):
    docs = request_documents()
    del docs["sketch"]["budgets"]["max_initial_synthesis_s"]
    docs["sketch"]["hyperparameters"]["total_solve_timeout_s"] = 10
    result = load(write_request(tmp_path, docs))
    assert result.request_config["sketch"]["budgets"]["max_initial_synthesis_s"] == 10
    assert result.value_sources["sketch.budgets.max_initial_synthesis_s"]["original_path"] == "sketch.hyperparameters.total_solve_timeout_s"
    docs["sketch"]["solver"]["total_solve_timeout_s"] = 10
    with pytest.raises(InputValidationError, match="duplicate"):
        load(write_request(tmp_path, docs))


def test_derived_reference_partition_has_correct_provenance(tmp_path):
    docs = request_documents()
    docs["atom"]["forbidden_transfers"] = [[0, 0, 1, 0]]
    result = load(write_request(tmp_path, docs))
    source = result.value_sources["sketch.hyperparameters.reference_slice_size_bytes"]
    assert source["kind"] == "derived"
    assert source["value"] == 1024


def test_cli_values_are_type_checked_before_conflict_comparison(tmp_path):
    docs = request_documents()
    files = write_request(tmp_path, docs)
    with pytest.raises(InputValidationError):
        load(files, {"sketch.collective.inplace": 0})


def test_huge_json_integer_reports_an_input_error(tmp_path):
    docs = request_documents()
    docs["sketch"]["hyperparameters"]["total_size_bytes"] = 10**400
    with pytest.raises(InputValidationError):
        load(write_request(tmp_path, docs))


def test_published_json_schemas_accept_null_defaults_and_reject_unknown_keys(tmp_path):
    from vericcl.input.schema import public_schemas
    from jsonschema import Draft202012Validator, ValidationError
    docs = request_documents()
    docs["sketch"]["collective"]["datatype"] = None
    docs["sketch"]["collective"]["root"] = None
    schemas = public_schemas()
    for name, schema in schemas.items():
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(docs[name])
    docs["atom"]["guess_a_method"] = True
    with pytest.raises(ValidationError):
        Draft202012Validator(schemas["atom"]).validate(docs["atom"])


@pytest.mark.parametrize("alias", ["ALL_REDUCE", "AllReduce"])
def test_cli_operator_aliases_are_normalized_before_conflict_checking(tmp_path, alias):
    files = write_request(tmp_path)
    assert load(files, {"sketch.collective.operator": alias}).collective.kind.value == "allreduce"


def test_successful_online_configuration_keeps_calibration_and_trace_independent(tmp_path):
    docs = request_documents()
    docs["sketch"]["calibration"] = {"enabled": True, "max_relative_fit_error": 0.1, "job_timeout_s": 5}
    docs["sketch"]["budgets"].update(max_online_validation_per_round_s=10, max_online_validation_total_s=30)
    result = load(write_request(tmp_path, docs))
    assert result.request_config["sketch"]["calibration"]["enabled"]
    assert not result.request_config["sketch"]["trace_and_measure"]["enabled"]


@pytest.mark.parametrize('field,value', [('pairs', 21), ('max_rounds', 4), ('max_cv', .1), ('min_relative_gain', 1)])
def test_paired_measurement_configuration_preserves_preplanned_contract(tmp_path, field, value):
    documents = request_documents()
    documents['sketch']['trace_and_measure'] = {field: value}
    with pytest.raises(InputValidationError):
        load(write_request(tmp_path, documents))
