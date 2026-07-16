import json
from pathlib import Path
from typing import Optional

import pytest

from vericcl.errors import InputValidationError
from vericcl.input.loader import resolve_inputs
from vericcl.input.models import ObjectiveMode
from vericcl.semantics.collective import CollectiveKind


pytestmark = pytest.mark.phase01


def write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_three_inputs(
    directory: Path,
    *,
    ranks: int = 2,
    total: int = 8_388_608,
    size: int = 1_048_576,
    operator: str = "allreduce",
    input_chunkup: Optional[int] = None,
):
    topology = {
        "name": "test-topology",
        "ranks": ranks,
        "directed_links": [],
        "nodes": [],
        "shared_resources": [],
    }
    sketch = {
        "collective": {
            "operator": operator,
            "root": None,
            "datatype": "float32",
            "reduction_op": (
                "sum"
                if operator in {"reduce", "allreduce", "reduce_scatter"}
                else None
            ),
        },
        "hyperparameters": {
            "total_size_bytes": total,
            "slice_size_bytes": size,
        },
    }
    if operator in {"broadcast", "reduce"}:
        sketch["collective"]["root"] = 0
    if input_chunkup is not None:
        sketch["hyperparameters"]["input_chunkup"] = input_chunkup
    atom = {
        "stage_num": None,
        "forbidden_transfers": [],
        "strategies": {},
        "manual_hierarchy": [],
    }
    return (
        write_json(directory / "topology.json", topology),
        write_json(directory / "sketch.json", sketch),
        write_json(directory / "atom.json", atom),
    )


def test_resolve_inputs_derives_global_rank_count_and_chunkup(tmp_path):
    paths = write_three_inputs(tmp_path)

    resolved = resolve_inputs(*paths)

    assert resolved.rank_count == 2
    assert resolved.hyperparameters.slice_count == 8
    assert resolved.collective.kind is CollectiveKind.ALL_REDUCE
    assert resolved.resolved_sketch["hyperparameters"]["input_chunkup"] == 8


def test_resolve_inputs_expands_all_defaults(tmp_path):
    paths = write_three_inputs(tmp_path)

    resolved = resolve_inputs(*paths)

    collective = resolved.resolved_sketch["collective"]
    hyperparameters = resolved.resolved_sketch["hyperparameters"]
    solver = resolved.resolved_sketch["solver"]
    strategies = resolved.resolved_atom["strategies"]
    assert collective["inplace"] is False
    assert hyperparameters["objective_mode"] == "auto"
    assert hyperparameters["max_calibration_channels"] == 32
    assert hyperparameters["min_expected_improvement"] == 0.01
    assert hyperparameters["min_tuning_improvement"] == 0.01
    assert hyperparameters["max_tuning_iterations"] == 20
    assert hyperparameters["total_verification_timeout_s"] == 10800
    assert hyperparameters["force_recalibrate"] is False
    assert solver["total_solve_timeout_s"] == 10800
    assert solver["per_model_timeout_s"] == 1800
    assert solver["mip_gap"] == 1e-4
    assert solver["solver_seed"] == 0
    assert strategies == {
        "batching": False,
        "constructive_trees": True,
        "hierarchy": False,
        "milp": True,
        "shortest_paths": False,
        "symmetry": False,
    }
    assert resolved.hyperparameters.objective_mode is ObjectiveMode.AUTO


def test_legacy_topology_derives_rank_count_from_node_geometry(tmp_path):
    paths = write_three_inputs(tmp_path)
    write_json(paths[0], {"name": "legacy", "nnodes": 2, "gpus_per_node": 4})

    resolved = resolve_inputs(*paths)

    assert resolved.rank_count == 8
    assert resolved.resolved_topology["ranks"] == 8


def test_atom_constraints_and_strategy_overrides_are_loaded(tmp_path):
    paths = write_three_inputs(tmp_path)
    write_json(
        paths[2],
        {
            "stage_num": 2,
            "forbidden_transfers": [[3, 0, 1, 1]],
            "strategies": {
                "hierarchy": True,
                "symmetry": True,
                "shortest_paths": True,
                "batching": False,
                "constructive_trees": True,
                "milp": False,
            },
            "manual_hierarchy": [{"stage_id": 0, "communication_group": [0, 1]}],
        },
    )

    resolved = resolve_inputs(*paths)

    forbidden = resolved.atom_constraints.forbidden_transfers[0]
    assert (forbidden.slice_id, forbidden.src_rank, forbidden.dst_rank) == (3, 0, 1)
    assert forbidden.stage_id == 1
    assert resolved.strategies.hierarchy is True
    assert resolved.strategies.batching is False
    assert resolved.strategies.milp is False
    assert resolved.strategies.manual_hierarchy[0]["stage_id"] == 0


@pytest.mark.parametrize("total,size", [(0, 1), (8, 0), (10, 4)])
def test_invalid_slice_geometry_is_rejected(tmp_path, total, size):
    paths = write_three_inputs(tmp_path, total=total, size=size)

    with pytest.raises(InputValidationError):
        resolve_inputs(*paths)


def test_inconsistent_input_chunkup_is_rejected(tmp_path):
    paths = write_three_inputs(tmp_path, input_chunkup=7)

    with pytest.raises(InputValidationError, match="input_chunkup"):
        resolve_inputs(*paths)


def test_reduce_scatter_requires_divisible_slice_count(tmp_path):
    paths = write_three_inputs(
        tmp_path,
        ranks=4,
        total=6,
        size=1,
        operator="reduce_scatter",
    )

    with pytest.raises(InputValidationError, match="slice count must be divisible"):
        resolve_inputs(*paths)


def test_resolved_hash_is_stable_for_reordered_input_keys(tmp_path):
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()
    left = write_three_inputs(left_dir)
    right = write_three_inputs(right_dir)
    topology = json.loads(right[0].read_text(encoding="utf-8"))
    sketch = json.loads(right[1].read_text(encoding="utf-8"))
    atom = json.loads(right[2].read_text(encoding="utf-8"))
    write_json(right[0], dict(reversed(tuple(topology.items()))))
    write_json(right[1], dict(reversed(tuple(sketch.items()))))
    write_json(right[2], dict(reversed(tuple(atom.items()))))

    assert resolve_inputs(*left).input_sha256 == resolve_inputs(*right).input_sha256


def test_missing_input_file_is_reported_as_input_error(tmp_path):
    paths = write_three_inputs(tmp_path)
    missing = tmp_path / "missing.json"

    with pytest.raises(InputValidationError, match="cannot read"):
        resolve_inputs(paths[0], paths[1], missing)


def test_packaged_examples_resolve():
    examples = Path(__file__).parents[3] / "vericcl" / "examples"

    resolved = resolve_inputs(
        examples / "topo" / "two_rank.json",
        examples / "sketch" / "allreduce_8m_1m.json",
        examples / "atom" / "default.json",
    )

    assert resolved.rank_count == 2
    assert resolved.hyperparameters.slice_count == 8


def test_unknown_atom_field_is_rejected(tmp_path):
    paths = write_three_inputs(tmp_path)
    write_json(paths[2], {"stage_count": 2})

    with pytest.raises(InputValidationError, match="unknown atom field"):
        resolve_inputs(*paths)


def test_unknown_collective_operator_is_rejected(tmp_path):
    paths = write_three_inputs(tmp_path, operator="scan")

    with pytest.raises(InputValidationError, match="unsupported collective operator"):
        resolve_inputs(*paths)


@pytest.mark.parametrize("operator", ["scatter", "gather"])
def test_internal_collectives_are_rejected_as_direct_inputs(tmp_path, operator):
    paths = write_three_inputs(tmp_path, operator=operator)

    with pytest.raises(InputValidationError, match="internal plan operator"):
        resolve_inputs(*paths)


def test_resolved_mappings_are_immutable(tmp_path):
    resolved = resolve_inputs(*write_three_inputs(tmp_path))

    with pytest.raises(TypeError):
        resolved.resolved_topology["ranks"] = 4


@pytest.mark.parametrize(
    "raw",
    [
        "[1, 2]",
        "{",
        '{"ranks": 2, "ranks": 3}',
        '{"ranks": NaN}',
    ],
)
def test_invalid_json_documents_are_rejected(tmp_path, raw):
    paths = write_three_inputs(tmp_path)
    paths[0].write_text(raw, encoding="utf-8")

    with pytest.raises(InputValidationError):
        resolve_inputs(*paths)


def test_topology_requires_rank_geometry(tmp_path):
    paths = write_three_inputs(tmp_path)
    write_json(paths[0], {"name": "missing-geometry"})

    with pytest.raises(InputValidationError, match="topology must define"):
        resolve_inputs(*paths)


@pytest.mark.parametrize(
    "collective,match",
    [
        (None, "collective is required"),
        ([], "must be a JSON object"),
        ({"datatype": "float32"}, "operator is required"),
        ({"operator": "allgather"}, "datatype is required"),
        (
            {"operator": "broadcast", "datatype": "float32", "root": "zero"},
            "must be an integer",
        ),
    ],
)
def test_invalid_collective_shapes_are_rejected(tmp_path, collective, match):
    paths = write_three_inputs(tmp_path)
    sketch = read_json(paths[1])
    if collective is None:
        del sketch["collective"]
    else:
        sketch["collective"] = collective
    write_json(paths[1], sketch)

    with pytest.raises(InputValidationError, match=match):
        resolve_inputs(*paths)


def test_missing_hyperparameters_are_rejected(tmp_path):
    paths = write_three_inputs(tmp_path)
    sketch = read_json(paths[1])
    del sketch["hyperparameters"]
    write_json(paths[1], sketch)

    with pytest.raises(InputValidationError, match="hyperparameters is required"):
        resolve_inputs(*paths)


def test_invalid_objective_mode_is_rejected(tmp_path):
    paths = write_three_inputs(tmp_path)
    sketch = read_json(paths[1])
    sketch["hyperparameters"]["objective_mode"] = "balanced"
    write_json(paths[1], sketch)

    with pytest.raises(InputValidationError, match="objective_mode"):
        resolve_inputs(*paths)


def test_legacy_solver_setting_in_hyperparameters_is_normalized(tmp_path):
    paths = write_three_inputs(tmp_path)
    sketch = read_json(paths[1])
    sketch["hyperparameters"]["solver_seed"] = 7
    write_json(paths[1], sketch)

    resolved = resolve_inputs(*paths)

    assert resolved.solver.solver_seed == 7
    assert resolved.resolved_sketch["solver"]["solver_seed"] == 7


def test_duplicate_solver_setting_is_rejected(tmp_path):
    paths = write_three_inputs(tmp_path)
    sketch = read_json(paths[1])
    sketch["hyperparameters"]["solver_seed"] = 7
    sketch["solver"] = {"solver_seed": 7}
    write_json(paths[1], sketch)

    with pytest.raises(InputValidationError, match="duplicate sketch setting"):
        resolve_inputs(*paths)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("hyperparameters", "max_calibration_channels", 33),
        ("hyperparameters", "min_expected_improvement", -0.1),
        ("hyperparameters", "min_tuning_improvement", 1.1),
        ("solver", "mip_gap", True),
        ("solver", "max_channels", 33),
    ],
)
def test_invalid_numeric_configuration_is_rejected(
    tmp_path,
    section,
    field,
    value,
):
    paths = write_three_inputs(tmp_path)
    sketch = read_json(paths[1])
    sketch.setdefault(section, {})[field] = value
    write_json(paths[1], sketch)

    with pytest.raises(InputValidationError):
        resolve_inputs(*paths)


def test_rank_geometry_in_sketch_is_rejected(tmp_path):
    paths = write_three_inputs(tmp_path)
    sketch = read_json(paths[1])
    sketch["ranks"] = 2
    write_json(paths[1], sketch)

    with pytest.raises(InputValidationError, match="belongs only in topology"):
        resolve_inputs(*paths)


@pytest.mark.parametrize(
    "atom,match",
    [
        ({"forbidden_transfers": {}}, "must be a list"),
        ({"forbidden_transfers": [[0, 1]]}, "must contain"),
        ({"forbidden_transfers": [[16, 0, 1, 0]]}, "slice_id is out of range"),
        ({"forbidden_transfers": [[0, 2, 1, 0]]}, "rank is out of range"),
        ({"forbidden_transfers": [[0, 0, 0, 0]]}, "distinct ranks"),
        (
            {"stage_num": 1, "forbidden_transfers": [[0, 0, 1, 1]]},
            "stage_id is out of range",
        ),
        ({"manual_hierarchy": {}}, "must be a list"),
        ({"manual_hierarchy": [1]}, "must be a JSON object"),
        (
            {"hierarchy": True, "strategies": {"hierarchy": True}},
            "duplicate atom strategy",
        ),
    ],
)
def test_invalid_atom_constraints_are_rejected(tmp_path, atom, match):
    paths = write_three_inputs(tmp_path)
    write_json(paths[2], atom)

    with pytest.raises(InputValidationError, match=match):
        resolve_inputs(*paths)
