from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable

from lxml import etree


PROJECT_ROOT = Path(__file__).parents[2]
EXAMPLES = PROJECT_ROOT / "vericcl" / "examples"

OFFLINE_DIMENSIONS = (
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
)
VALIDATION_DIMENSIONS = OFFLINE_DIMENSIONS + ("runtime", "online")


def write_inputs(
    directory: Path,
    operator: str,
    *,
    inplace: bool = False,
    topology_name: str = "two_rank.json",
    topology_path: Path | None = None,
    total_size_bytes: int = 2048,
    slice_size_bytes: int = 1024,
    max_channels: int = 1,
    hierarchy: bool = False,
    constructive_trees: bool = True,
    milp: bool = False,
    max_parallel_models: int = 1,
    max_threads_per_model: int = 1,
    total_solve_timeout_s: int = 300,
    per_model_timeout_s: int = 60,
) -> tuple[Path, Path, Path]:
    topology = topology_path or EXAMPLES / "topo" / topology_name
    sketch_payload = json.loads(
        (EXAMPLES / "sketch" / "allreduce_8m_1m.json").read_text(
            encoding="utf-8"
        )
    )
    reduction = operator in {"reduce", "allreduce", "reducescatter"}
    rooted = operator in {"broadcast", "reduce"}
    sketch_payload["collective"] = {
        "operator": operator,
        "root": 0 if rooted else None,
        "datatype": "float32",
        "reduction_op": "sum" if reduction else None,
        "inplace": inplace,
    }
    sketch_payload["hyperparameters"].update(
        {
            "total_size_bytes": total_size_bytes,
            "slice_size_bytes": slice_size_bytes,
            "input_chunkup": total_size_bytes // slice_size_bytes,
            "objective_mode": "latency",
            "max_tuning_iterations": 1,
        }
    )
    sketch_payload["solver"].update(
        {
            "solver_seed": 0,
            "max_channels": max_channels,
            "max_parallel_models": max_parallel_models,
            "max_threads_per_model": max_threads_per_model,
            "total_solve_timeout_s": total_solve_timeout_s,
            "per_model_timeout_s": per_model_timeout_s,
        }
    )
    atom_payload = json.loads(
        (EXAMPLES / "atom" / "default.json").read_text(encoding="utf-8")
    )
    atom_payload["strategies"].update(
        {
            "hierarchy": hierarchy,
            "constructive_trees": constructive_trees,
            "milp": milp,
        }
    )
    sketch = directory / "sketch.json"
    atom = directory / "atom.json"
    sketch.write_text(json.dumps(sketch_payload), encoding="utf-8")
    atom.write_text(json.dumps(atom_payload), encoding="utf-8")
    return topology, sketch, atom


def solve_public_cli(
    directory: Path,
    operator: str,
    *,
    inplace: bool = False,
    topology_name: str = "two_rank.json",
    topology_path: Path | None = None,
    total_size_bytes: int = 2048,
    slice_size_bytes: int = 1024,
    max_channels: int = 1,
    hierarchy: bool = False,
    constructive_trees: bool = True,
    milp: bool = False,
    max_parallel_models: int = 1,
    max_threads_per_model: int = 1,
    total_solve_timeout_s: int = 300,
    per_model_timeout_s: int = 60,
    command_timeout_s: int = 300,
    run_id: str = "acceptance",
) -> dict[str, object]:
    inputs_dir = directory / "inputs"
    inputs_dir.mkdir(parents=True)
    topology, sketch, atom = write_inputs(
        inputs_dir,
        operator,
        inplace=inplace,
        topology_name=topology_name,
        topology_path=topology_path,
        total_size_bytes=total_size_bytes,
        slice_size_bytes=slice_size_bytes,
        max_channels=max_channels,
        hierarchy=hierarchy,
        constructive_trees=constructive_trees,
        milp=milp,
        max_parallel_models=max_parallel_models,
        max_threads_per_model=max_threads_per_model,
        total_solve_timeout_s=total_solve_timeout_s,
        per_model_timeout_s=per_model_timeout_s,
    )
    output_dir = directory / "runs"
    command = (
        sys.executable,
        "-m",
        "vericcl",
        "solve",
        "--topology",
        str(topology),
        "--sketch",
        str(sketch),
        "--atoms",
        str(atom),
        "--output-dir",
        str(output_dir),
        "--run-id",
        run_id,
    )
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=command_timeout_s,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "public CLI failed with code {}:\nstdout={}\nstderr={}".format(
                completed.returncode,
                completed.stdout,
                completed.stderr,
            )
        )
    roots = tuple(path for path in output_dir.iterdir() if path.is_dir())
    assert len(roots) == 1
    root = roots[0]
    final_xmls = tuple(root.glob("*_final*.xml"))
    final_reports = tuple(root.glob("*_final.validation.json"))
    final_sidecars = tuple(root.glob("*_final.schedule.json"))
    assert len(final_xmls) == len(final_reports) == len(final_sidecars) == 1
    return {
        "root": root,
        "xml_path": final_xmls[0],
        "report_path": final_reports[0],
        "sidecar_path": final_sidecars[0],
        "summary": json.loads(
            (root / "run-summary.json").read_text(encoding="utf-8")
        ),
        "report": json.loads(final_reports[0].read_text(encoding="utf-8")),
        "sidecar": json.loads(final_sidecars[0].read_text(encoding="utf-8")),
        "xml": etree.parse(str(final_xmls[0])).getroot(),
        "stdout": completed.stdout,
    }


def expected_outputs(
    operator: str,
    rank_count: int,
    slice_count: int,
    root: int = 0,
) -> dict[str, list[int]]:
    outputs: dict[str, list[int]] = {}
    if operator == "broadcast":
        for rank in range(rank_count):
            for logical in range(slice_count):
                outputs[slot(rank, logical)] = [root * slice_count + logical]
    elif operator == "reduce":
        for logical in range(slice_count):
            outputs[slot(root, logical)] = [
                rank * slice_count + logical for rank in range(rank_count)
            ]
    elif operator == "allgather":
        for rank in range(rank_count):
            for source in range(rank_count):
                for logical in range(slice_count):
                    offset = source * slice_count + logical
                    outputs[slot(rank, offset)] = [offset]
    elif operator == "allreduce":
        for rank in range(rank_count):
            for logical in range(slice_count):
                outputs[slot(rank, logical)] = [
                    source * slice_count + logical
                    for source in range(rank_count)
                ]
    elif operator == "alltoall":
        per_destination = slice_count // rank_count
        for source in range(rank_count):
            for logical in range(slice_count):
                rank = logical // per_destination
                offset = source * per_destination + logical % per_destination
                outputs[slot(rank, offset)] = [
                    source * slice_count + logical
                ]
    elif operator == "reducescatter":
        per_destination = slice_count // rank_count
        for logical in range(slice_count):
            rank = logical // per_destination
            offset = logical % per_destination
            outputs[slot(rank, offset)] = [
                source * slice_count + logical
                for source in range(rank_count)
            ]
    else:
        raise AssertionError("unsupported operator: {}".format(operator))
    return outputs


def slot(rank: int, offset: int) -> str:
    return "r{:08d}-o{:08d}".format(rank, offset)


def assert_semantic_outputs(result: dict[str, object], operator: str) -> None:
    sidecar = result["sidecar"]
    schedule = sidecar["schedule"]
    actual = schedule["metadata"]["final_outputs"]
    expected = expected_outputs(
        operator,
        schedule["rank_count"],
        schedule["slice_count"],
    )
    assert actual == expected


def assert_validation_report(result: dict[str, object]) -> None:
    report = result["report"]
    assert set(report["validation"]) == set(VALIDATION_DIMENSIONS)
    for dimension in OFFLINE_DIMENSIONS:
        assert report["validation"][dimension]["status"] == "valid"
    assert report["validation"]["runtime"]["status"] == "valid"
    assert report["validation"]["online"]["status"] == "not_run"


def _rank_chunks(report: dict, kind: str, rank: int) -> int:
    return report["buffer_plan"][kind]["r{:08d}".format(rank)]


def assert_xml_contract(result: dict[str, object]) -> None:
    root = result["xml"]
    report = result["report"]
    sidecar = result["sidecar"]["schedule"]
    assert int(root.attrib["ngpus"]) == sidecar["rank_count"]
    expected_loop_chunks = sidecar["slice_count"]
    if root.attrib["coll"] == "allgather":
        expected_loop_chunks *= sidecar["rank_count"]
    assert int(root.attrib["nchunksperloop"]) == expected_loop_chunks
    sends: Counter[tuple[int, int, int]] = Counter()
    receives: Counter[tuple[int, int, int]] = Counter()
    chunks_by_rank = {
        int(gpu.attrib["id"]): {
            "i_chunks": int(gpu.attrib["i_chunks"]),
            "o_chunks": int(gpu.attrib["o_chunks"]),
            "s_chunks": int(gpu.attrib["s_chunks"]),
        }
        for gpu in root.xpath("./gpu")
    }
    for gpu in root.xpath("./gpu"):
        rank = int(gpu.attrib["id"])
        expected_chunks = {
            "i_chunks": _rank_chunks(report, "input_chunks", rank),
            "o_chunks": _rank_chunks(report, "output_chunks", rank),
            "s_chunks": _rank_chunks(report, "scratch_chunks", rank),
        }
        assert {key: int(gpu.attrib[key]) for key in expected_chunks} == (
            expected_chunks
        )
        for tb in gpu.xpath("./tb"):
            channel = int(tb.attrib["chan"])
            peer_send = int(tb.attrib["send"])
            peer_recv = int(tb.attrib["recv"])
            steps = tb.xpath("./step")
            assert [int(step.attrib["s"]) for step in steps] == list(
                range(len(steps))
            )
            for step in steps:
                assert int(step.attrib["cnt"]) == 1
                step_type = step.attrib["type"]
                if step_type == "nop":
                    continue
                reference_ranks = {
                    "s": (rank, peer_send),
                    "r": (peer_recv, rank),
                    "rrc": (rank, rank),
                    "cpy": (rank, rank),
                }[step_type]
                for prefix, reference_rank in zip(
                    ("src", "dst"), reference_ranks
                ):
                    buffer_name = step.attrib[prefix + "buf"]
                    buffer_key = {
                        "i": "i_chunks",
                        "o": "o_chunks",
                        "s": "s_chunks",
                    }[buffer_name]
                    offset = int(step.attrib[prefix + "off"])
                    assert (
                        0
                        <= offset
                        < chunks_by_rank[reference_rank][buffer_key]
                    )
                if step_type == "s":
                    assert peer_send >= 0 and peer_recv == -1
                    sends[(rank, peer_send, channel)] += 1
                if step_type in {"r", "rrc"}:
                    assert peer_recv >= 0 and peer_send == -1
                    receives[(peer_recv, rank, channel)] += 1
    assert sends == receives
    lanes: dict[tuple[int, int, int], list[tuple[float, float]]] = defaultdict(
        list
    )
    for transfer in sidecar["transfers"]:
        lane = (
            transfer["src_rank"],
            transfer["dst_rank"],
            transfer["channel"],
        )
        lanes[lane].append((transfer["st_time"], transfer["ed_time"]))
    for intervals in lanes.values():
        ordered = sorted(intervals)
        assert all(
            left[1] <= right[0]
            for left, right in zip(ordered, ordered[1:])
        )


def assert_exact_tiny_buffers(
    result: dict[str, object],
    operator: str,
    inplace: bool,
) -> None:
    expected = {
        "broadcast": (2, 2, 0, 0 if inplace else 2),
        "reduce": (2, 2, 0, 0 if inplace else 2),
        "allgather": (2, 4, 0, 0 if inplace else 4),
        "allreduce": (2, 2, 0, 0 if inplace else 2),
        "alltoall": (2, 2, 1 if inplace else 0, 2),
        "reducescatter": (2, 1, 0, 0 if inplace else 2),
    }[operator]
    plan = result["report"]["buffer_plan"]
    for field, value in zip(
        ("input_chunks", "output_chunks", "scratch_chunks"),
        expected[:3],
    ):
        assert set(plan[field].values()) == {value}
    assert plan["local_copy_count"] == expected[3]


def canonical_report_sections(report: dict) -> dict:
    return {
        key: report[key]
        for key in (
            "normalized_input_sha256",
            "topology_signature",
            "candidate_signature",
            "artifact_binding_sha256",
            "requested_strategies",
            "applied_strategies",
            "strategy_parameters",
            "hierarchy_plan",
            "channel_count",
            "buffer_plan",
            "solver_metrics",
            "validation",
            "proven_optimal",
            "search_space_restricted",
            "runtime_compatible",
            "xml_sha256",
            "reproducibility",
        )
    }


def transfer_pairs(transfers: Iterable[dict]) -> set[tuple[int, int]]:
    return {
        (transfer["src_rank"], transfer["dst_rank"])
        for transfer in transfers
    }


def write_multi_rail_topology(path: Path) -> Path:
    node_ranks = ((0, 1, 2, 3), (4, 5, 6, 7))
    rail_pairs = ((0, 4), (1, 5), (2, 6), (3, 7))
    links = []
    for ranks in node_ranks:
        for src_rank in ranks:
            for dst_rank in ranks:
                if src_rank == dst_rank:
                    continue
                links.append(
                    {
                        "src": src_rank,
                        "dst": dst_rank,
                        "max_channels": 4,
                        "alpha": 1.0,
                        "beta": 1.0,
                        "invbw": 2.0,
                        "resources": [],
                    }
                )
    for left_rank, right_rank in rail_pairs:
        for src_rank, dst_rank in (
            (left_rank, right_rank),
            (right_rank, left_rank),
        ):
            links.append(
                {
                    "src": src_rank,
                    "dst": dst_rank,
                    "max_channels": 4,
                    "alpha": 2.0,
                    "beta": 3.0,
                    "invbw": 5.0,
                    "resources": [],
                }
            )
    payload = {
        "name": "two-node-four-rail",
        "ranks": 8,
        "nodes": [
            {"id": 0, "ranks": list(node_ranks[0]), "gateways": [0, 1, 2, 3]},
            {"id": 1, "ranks": list(node_ranks[1]), "gateways": [4, 5, 6, 7]},
        ],
        "directed_links": links,
        "shared_resources": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def assert_no_global_stage_barrier(result: dict[str, object]) -> None:
    schedule = result["sidecar"]["schedule"]
    assert schedule["metadata"].get("stage_barrier") is None
    by_stage: dict[int, list[dict]] = defaultdict(list)
    for transfer in schedule["transfers"]:
        by_stage[transfer["stage_id"]].append(transfer)
    ordered_stages = sorted(by_stage)
    assert len(ordered_stages) > 1
    for left_stage, right_stage in zip(ordered_stages, ordered_stages[1:]):
        assert min(
            transfer["st_time"] for transfer in by_stage[right_stage]
        ) < max(transfer["ed_time"] for transfer in by_stage[left_stage])


def assert_reduction_atoms(result: dict[str, object]) -> None:
    transfers = result["sidecar"]["schedule"]["transfers"]
    reduced = [transfer for transfer in transfers if transfer["kind"] == "REDUCE"]
    assert reduced
    for transfer in reduced:
        assert set(transfer["member_slice_ids"]) == {
            atom["slice_id"] for atom in transfer["atoms"]
        }


def assert_cross_stage_accumulator_dependencies(
    result: dict[str, object],
) -> None:
    schedule = result["sidecar"]["schedule"]
    by_id = {
        transfer["transfer_id"]: transfer
        for transfer in schedule["transfers"]
    }
    semantic = schedule["metadata"]["semantic_predecessors"]
    cross_stage_reductions = []
    for transfer in schedule["transfers"]:
        if transfer["kind"] != "REDUCE":
            continue
        earlier = [
            by_id[predecessor_id]
            for predecessor_id in semantic[transfer["transfer_id"]]
            if by_id[predecessor_id]["stage_id"] < transfer["stage_id"]
        ]
        if not earlier:
            continue
        cross_stage_reductions.append(transfer)
        assert any(
            predecessor["dst_rank"] == transfer["dst_rank"]
            for predecessor in earlier
        )
    assert cross_stage_reductions
