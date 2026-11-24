"""Convert the legacy V100/A100 experiment fixtures to v5 request documents.

Usage (from the repository root)::

    .venv/bin/python tools/convert_legacy_experiments.py \
        [--legacy-root tests/fixtures/legacy-experiments] \
        [--legacy-atom tests/fixtures/legacy-catalog/atom/default.json] \
        [--output tests/fixtures/experiments-v5]

Output layout (all paths repository-relative)::

    <output>/atom.json                         shared v5 atom
    <output>/topology/<topology>.json          v5 topology
    <output>/sketch/<topology>/<op>-<size>.json
    <output>/runtime/<topology>/<op>-<size>.json
    <output>/<group>-k16-manifest.json         schema_version 2 manifests

Mapping of the legacy experimental meaning onto v5:

* Topology: ``alpha`` is kept as ``alpha_us``. Legacy ``beta`` was a per-slice
  duration independent of the slice size (``invbw = alpha + beta``); v5 needs a
  per-byte cost, so ``beta_us_per_byte = beta / --beta-reference-bytes``
  (default 64 KiB, i.e. intra-node 1 us / 64 KiB and inter-node 12 us /
  64 KiB keep their 12x ratio). Shared resources become throughput caps with
  ``capacity_bytes_per_us = reference / beta`` over the same member links (v5
  shared resources add no startup delay). ``max_channels`` is clamped to the
  v5 maximum of 16, which equals every legacy solver channel limit.
* Sketch: collective, datatype, inplace, reduction, total and slice sizes are
  copied (``input_chunkup`` is checked against total/slice and dropped, since
  v5 derives the slice count). Solver limits are copied; ``cpu_budget`` is
  ``max_threads_per_model * max_parallel_models`` and ``channel_candidates``
  enumerates 1..max_channels. Timeouts become budgets: the legacy solve
  timeout is ``max_initial_synthesis_s``; the legacy verification timeout caps
  offline validation, online validation and repair (legacy had per-phase
  totals only, so per-round caps equal the totals);
  ``max_tuning_iterations`` is ``max_tuning_rounds``. Legacy solves always ran
  online validation and tuning, so calibration and trace-and-measure are
  enabled and ``min_tuning_improvement`` is the measured
  ``min_relative_gain``. Settings v5 requires but legacy never had come from
  the V100 harness (:mod:`vericcl.experiments.v100`): the fit-error limit,
  the per-job timeout of calibration and trace jobs and the cleanup reserve
  are the values of the two-server tuning example inputs. The trace
  buffer holds one traced invocation at MSCCL's per-rank maximum (216
  threadblocks x 256 steps, one record per step), so it never truncates a
  valid XML. ``min_expected_improvement``, ``force_resolve`` and
  ``force_recalibrate`` have no v5 counterpart (v5 never reuses schedules or
  calibrations unless ``sketch.history`` enables it).
* Atom: legacy strategy booleans select ``synthesis_method``; the
  ``objective_mode`` (identical in every legacy sketch) moves to the atom.
* Runtime: a host-independent v5 runtime per case. Ranks of topology node
  ``i`` run on the logical host ``worker-<letter i>`` (device = index within
  the node). ``NCCL_BUFFSIZE`` is explicit and keeps the legacy harness value
  of twice the slice size; ``scratch_bytes_per_rank`` is the collective's
  per-rank output size. The V100 harness binds logical hosts, tool paths and
  deployment environment from its config (see vericcl/experiments/v100.py).

Every converted case is validated with ``vericcl.input.loader.load_request``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import string
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vericcl.experiments.v100 import (  # noqa: E402
    CALIBRATION_MAX_RELATIVE_FIT_ERROR,
    CLEANUP_RESERVE_S,
    MEASUREMENT_JOB_TIMEOUT_S,
)
from vericcl.verification.online.trace_format import RAW_HEADER_STRUCT, RAW_RECORD_STRUCT  # noqa: E402
from vericcl.xml.capacity import RuntimeLimits  # noqa: E402

SCHEMA_MAX_CHANNELS = 16
DEFAULT_BETA_REFERENCE_BYTES = 64 * 1024
COLLECTIVE_LABELS = {"allgather": "ag", "allreduce": "ar"}
DATATYPE_BYTES = {"float16": 2, "float32": 4, "float64": 8, "bfloat16": 2,
                  "int8": 1, "uint8": 1, "int32": 4, "uint32": 4, "int64": 8, "uint64": 8}

# Measurement settings for the online stages the legacy harness always ran.
CALIBRATION = {"enabled": True, "max_relative_fit_error": CALIBRATION_MAX_RELATIVE_FIT_ERROR,
               "job_timeout_s": MEASUREMENT_JOB_TIMEOUT_S}
TRACE_JOB_TIMEOUT_S = MEASUREMENT_JOB_TIMEOUT_S
_LIMITS = RuntimeLimits()
TRACE_CAPACITY_BYTES = RAW_HEADER_STRUCT.size + RAW_RECORD_STRUCT.size * _LIMITS.tbs_per_rank * _LIMITS.steps_per_tb
# The comparison tolerance every shipped v5 example uses.
TOLERANCES = {"absolute": 1e-5, "relative": 1e-5}


class ConversionError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise ConversionError(message)


def _exact_keys(value, keys, label):
    _require(isinstance(value, dict) and set(value) == set(keys),
             "{} has unexpected fields: {}".format(label, sorted(set(value) ^ set(keys)) if isinstance(value, dict) else value))


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def _relative(path):
    return Path(path).resolve().relative_to(REPO_ROOT).as_posix()


def _number(value):
    return int(value) if float(value).is_integer() else value


def convert_topology(legacy, reference_bytes):
    _require(set(legacy) <= {"name", "ranks", "nodes", "directed_links", "shared_resources"},
             "legacy topology has unsupported fields")
    links = []
    members = {}
    for edge in legacy["directed_links"]:
        _exact_keys(edge, ("src", "dst", "max_channels", "alpha", "beta", "invbw", "resources"), "legacy link")
        _require(abs(edge["alpha"] + edge["beta"] - edge["invbw"]) <= 1e-9 * max(1.0, edge["invbw"]),
                 "legacy link invbw differs from alpha + beta")
        _require(edge["beta"] > 0, "legacy link beta must be positive")
        links.append({
            "src": edge["src"], "dst": edge["dst"],
            "alpha_us": _number(edge["alpha"]),
            "beta_us_per_byte": edge["beta"] / reference_bytes,
            "max_channels": min(edge["max_channels"], SCHEMA_MAX_CHANNELS),
            "resources": list(edge["resources"]),
        })
        for resource in edge["resources"]:
            members.setdefault(resource, set()).add((edge["src"], edge["dst"]))
    resources = []
    for resource in legacy.get("shared_resources", ()):
        _exact_keys(resource, ("id", "member_links", "max_channels", "alpha", "beta", "invbw"), "legacy resource")
        declared = {tuple(pair) for pair in resource["member_links"]}
        _require(declared == members.get(resource["id"], set()),
                 "legacy resource {} members differ from its links".format(resource["id"]))
        resources.append({
            "id": resource["id"],
            "kind": "nic" if resource["id"].startswith("nic-") else "link",
            "capacity_bytes_per_us": reference_bytes / resource["beta"],
            "links": [list(pair) for pair in sorted(declared)],
        })
    _require(set(members) <= {item["id"] for item in resources}, "link references an undeclared resource")
    return {
        "name": legacy["name"],
        "ranks": legacy["ranks"],
        "nodes": [{"id": node["id"], "ranks": list(node["ranks"]), "gateways": list(node.get("gateways", ()))}
                  for node in legacy["nodes"]],
        "directed_links": links,
        "shared_resources": resources,
    }


def convert_atom(legacy, objective_mode):
    _exact_keys(legacy, ("stage_num", "forbidden_transfers", "strategies", "manual_hierarchy"), "legacy atom")
    strategies = legacy["strategies"]
    _exact_keys(strategies, ("hierarchy", "symmetry", "shortest_paths", "batching", "constructive_trees", "milp"),
                "legacy atom strategies")
    _require(strategies["milp"], "v5 has no non-MILP synthesis")
    _require(strategies["hierarchy"] != strategies["constructive_trees"],
             "legacy atom must select exactly one of hierarchy and constructive_trees")
    _require(not legacy["forbidden_transfers"] and not legacy["manual_hierarchy"],
             "legacy forbidden transfers and manual hierarchies are not converted")
    atom = {
        "synthesis_method": "hierarchy" if strategies["hierarchy"] else "constructive_trees",
        "strategies": {"symmetry": strategies["symmetry"], "shortest_paths": strategies["shortest_paths"],
                       "batching": strategies["batching"], "objective_mode": objective_mode},
    }
    if legacy["stage_num"] is not None:
        atom["stage_num"] = legacy["stage_num"]
    return atom


_HYPERPARAMETERS = ("total_size_bytes", "slice_size_bytes", "input_chunkup", "objective_mode",
                    "max_calibration_channels", "min_expected_improvement", "min_tuning_improvement",
                    "max_tuning_iterations", "total_verification_timeout_s", "force_recalibrate")
_SOLVER = ("total_solve_timeout_s", "per_model_timeout_s", "mip_gap", "require_proven_optimal", "solver_seed",
           "max_channels", "max_threads_per_model", "max_parallel_models", "force_resolve")


def convert_sketch(legacy):
    _exact_keys(legacy, ("collective", "hyperparameters", "solver"), "legacy sketch")
    collective, hyper, solver = legacy["collective"], legacy["hyperparameters"], legacy["solver"]
    _exact_keys(collective, ("operator", "root", "datatype", "reduction_op", "inplace"), "legacy collective")
    _exact_keys(hyper, _HYPERPARAMETERS, "legacy hyperparameters")
    _exact_keys(solver, _SOLVER, "legacy solver")
    total, slice_size = hyper["total_size_bytes"], hyper["slice_size_bytes"]
    _require(total % slice_size == 0 and total // slice_size == hyper["input_chunkup"],
             "legacy input_chunkup differs from total/slice")
    _require(hyper["max_calibration_channels"] == solver["max_channels"],
             "v5 calibrates up to solver.max_channels")
    _require(solver["max_channels"] <= SCHEMA_MAX_CHANNELS, "channel limit exceeds the v5 maximum")
    converted_collective = {"operator": collective["operator"], "datatype": collective["datatype"],
                            "inplace": collective["inplace"]}
    for key in ("root", "reduction_op"):
        if collective[key] is not None:
            converted_collective[key] = collective[key]
    verification = hyper["total_verification_timeout_s"]
    return {
        "collective": converted_collective,
        "hyperparameters": {"total_size_bytes": total, "slice_size_bytes": slice_size},
        "solver": {
            "backend": "scip",
            "per_model_timeout_s": solver["per_model_timeout_s"],
            "cpu_budget": solver["max_threads_per_model"] * solver["max_parallel_models"],
            "max_threads_per_model": solver["max_threads_per_model"],
            "max_parallel_models": solver["max_parallel_models"],
            "max_channels": solver["max_channels"],
            "channel_candidates": list(range(1, solver["max_channels"] + 1)),
            "solver_seed": solver["solver_seed"],
            "mip_gap": solver["mip_gap"],
            "require_proven_optimal": solver["require_proven_optimal"],
        },
        "budgets": {
            "max_initial_synthesis_s": solver["total_solve_timeout_s"],
            "max_offline_validation_per_round_s": verification,
            "max_offline_validation_total_s": verification,
            "max_online_validation_per_round_s": verification,
            "max_online_validation_total_s": verification,
            "max_repair_per_round_s": verification,
            "max_repair_total_s": verification,
            "max_tuning_rounds": hyper["max_tuning_iterations"],
            "cleanup_reserve_s": CLEANUP_RESERVE_S,
        },
        "tolerances": dict(TOLERANCES),
        "calibration": dict(CALIBRATION),
        "trace_and_measure": {
            "enabled": True,
            "min_relative_gain": hyper["min_tuning_improvement"],
            "job_timeout_s": TRACE_JOB_TIMEOUT_S,
            "trace_capacity_bytes": TRACE_CAPACITY_BYTES,
        },
    }


def logical_host(node_index):
    return "worker-" + string.ascii_lowercase[node_index]


def build_runtime(topology, sketch):
    rank_map = []
    for index, node in enumerate(sorted(topology["nodes"], key=lambda item: item["id"])):
        for device, rank in enumerate(sorted(node["ranks"])):
            rank_map.append({"rank": rank, "host": logical_host(index), "device": device})
    total = sketch["hyperparameters"]["total_size_bytes"]
    slice_size = sketch["hyperparameters"]["slice_size_bytes"]
    output = total * topology["ranks"] if sketch["collective"]["operator"] == "allgather" else total
    buffsize = 2 * slice_size
    _require(0 < buffsize <= 2**31 - 1, "NCCL_BUFFSIZE exceeds a signed 32-bit byte count")
    return {
        "rank_map": sorted(rank_map, key=lambda item: item["rank"]),
        "environment": {"NCCL_BUFFSIZE": str(buffsize), "NCCL_PROTO": "Simple"},
        "protocol": "Simple",
        "tb_layout": "separate",
        "scratch_bytes_per_rank": output,
        "device_memory_budget_bytes": 0,
    }


def _size_label(sketch_path):
    operator_label, _, size_label = sketch_path.stem.partition("-")
    _require(size_label, "legacy sketch name must be <op>-<size>.json")
    return operator_label, size_label


def convert(legacy_root, legacy_atom, output, reference_bytes):
    legacy_root, output = Path(legacy_root).resolve(), Path(output).resolve()
    manifests = sorted(legacy_root.glob("*-manifest.json"))
    _require(manifests, "no legacy manifest found")
    legacy_manifests = {path.name: _read(path) for path in manifests}
    for name, manifest in legacy_manifests.items():
        _exact_keys(manifest, ("schema_version", "atom", "topologies", "collectives", "sizes"), name)
        _require(manifest["schema_version"] == 1, name + " is not a legacy manifest")
    topologies = sorted(path.stem for path in (legacy_root / "topo").glob("*.json"))
    covered = {name for manifest in legacy_manifests.values() for name in manifest["topologies"]}
    reference = legacy_manifests[manifests[0].name]
    groups = {name: list(manifest["topologies"]) for name, manifest in legacy_manifests.items()}
    for name in topologies:
        if name not in covered:
            groups.setdefault("{}-k16-manifest.json".format(name.split("-")[0]), []).append(name)

    objective_modes = set()
    cases = {}
    for topology_name in topologies:
        legacy_topology = _read(legacy_root / "topo" / (topology_name + ".json"))
        _require(legacy_topology["name"] == topology_name, "legacy topology name differs from its file")
        topology = convert_topology(legacy_topology, reference_bytes)
        topology_path = output / "topology" / (topology_name + ".json")
        _write(topology_path, topology)
        for sketch_path in sorted((legacy_root / "sketch" / topology_name).glob("*/*.json")):
            legacy = _read(sketch_path)
            label, size_label = _size_label(sketch_path)
            _require(COLLECTIVE_LABELS.get(legacy["collective"]["operator"]) == label == sketch_path.parent.name,
                     "legacy sketch collective differs from its path: " + str(sketch_path))
            objective_modes.add(legacy["hyperparameters"]["objective_mode"])
            sketch = convert_sketch(legacy)
            runtime = build_runtime(topology, sketch)
            stem = "{}-{}".format(label, size_label)
            sketch_out = output / "sketch" / topology_name / (stem + ".json")
            runtime_out = output / "runtime" / topology_name / (stem + ".json")
            _write(sketch_out, sketch)
            _write(runtime_out, runtime)
            cases[(topology_name, label, size_label)] = {
                "topology_name": topology_name, "collective_label": label, "size_label": size_label,
                "topology": _relative(topology_path), "sketch": _relative(sketch_out),
                "runtime": _relative(runtime_out)}
    _require(len(objective_modes) == 1, "legacy sketches disagree on objective_mode; the v5 atom is shared")
    atom_path = output / "atom.json"
    _write(atom_path, convert_atom(_read(legacy_atom), objective_modes.pop()))

    written = []
    for name, members in sorted(groups.items()):
        source = legacy_manifests.get(name, reference)
        selected = [cases[(topology_name, label, size)] for topology_name in members
                    for label in source["collectives"] for size in source["sizes"]]
        path = output / name
        _write(path, {"schema_version": 2, "atom": _relative(atom_path), "cases": selected})
        written.append(path)
    return atom_path, cases, written


def validate(atom_path, cases):
    from vericcl.input.loader import load_request
    for case in cases.values():
        files = {"topology": REPO_ROOT / case["topology"], "sketch": REPO_ROOT / case["sketch"],
                 "atom": atom_path, "runtime": REPO_ROOT / case["runtime"]}
        for purpose in ("solve", "calibrate"):
            load_request(files, purpose=purpose)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--legacy-root", type=Path, default=REPO_ROOT / "tests/fixtures/legacy-experiments")
    parser.add_argument("--legacy-atom", type=Path, default=REPO_ROOT / "tests/fixtures/legacy-catalog/atom/default.json")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "tests/fixtures/experiments-v5")
    parser.add_argument("--beta-reference-bytes", type=int, default=DEFAULT_BETA_REFERENCE_BYTES)
    args = parser.parse_args(argv)
    _require(args.beta_reference_bytes > 0, "--beta-reference-bytes must be positive")
    atom_path, cases, manifests = convert(args.legacy_root, args.legacy_atom, args.output, args.beta_reference_bytes)
    validate(atom_path, cases)
    print("converted {} cases; manifests: {}".format(
        len(cases), ", ".join(_relative(path) for path in manifests)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
