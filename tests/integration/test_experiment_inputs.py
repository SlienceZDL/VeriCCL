import json
from pathlib import Path

from vericcl.input.loader import load_request
from vericcl.planner import build_plan
from vericcl.topology.loader import load_topology


ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests/fixtures/experiments-v5"


def test_all_experiment_inputs_resolve_and_build():
    cases = []
    for manifest_path in sorted(FIXTURES.glob("*-manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        assert manifest["schema_version"] == 2
        for case in manifest["cases"]:
            files = {name: ROOT / case[name] for name in ("topology", "sketch", "runtime")}
            files["atom"] = ROOT / manifest["atom"]
            for purpose in ("solve", "calibrate"):
                resolved = load_request(files, purpose=purpose)
            assert resolved.solver.max_channels == 16
            assert resolved.hyperparameters.max_calibration_channels == 16
            sketch = resolved.request_config["sketch"]
            assert sketch["calibration"]["enabled"] and sketch["trace_and_measure"]["enabled"]
            build_plan(resolved, load_topology(resolved))
            cases.append((case["topology"], case["sketch"]))
    assert len(cases) == len(set(cases)) == 36


def test_converted_fixtures_preserve_legacy_meaning():
    legacy_root = ROOT / "tests/fixtures/legacy-experiments"
    for legacy_path in sorted((legacy_root / "sketch").glob("*/*/*.json")):
        legacy = json.loads(legacy_path.read_text())
        topology = legacy_path.parents[1].name
        sketch = json.loads((FIXTURES / "sketch" / topology / legacy_path.name).read_text())
        runtime = json.loads((FIXTURES / "runtime" / topology / legacy_path.name).read_text())
        collective = legacy["collective"]
        assert sketch["collective"]["operator"] == collective["operator"]
        assert sketch["collective"]["datatype"] == collective["datatype"]
        assert sketch["collective"]["inplace"] == collective["inplace"]
        assert sketch["collective"].get("reduction_op") == collective["reduction_op"]
        for key in ("total_size_bytes", "slice_size_bytes"):
            assert sketch["hyperparameters"][key] == legacy["hyperparameters"][key]
        for key in ("per_model_timeout_s", "max_channels", "max_threads_per_model",
                    "max_parallel_models", "mip_gap", "solver_seed", "require_proven_optimal"):
            assert sketch["solver"][key] == legacy["solver"][key]
        budgets = sketch["budgets"]
        assert budgets["max_initial_synthesis_s"] == legacy["solver"]["total_solve_timeout_s"]
        assert budgets["max_offline_validation_total_s"] == legacy["hyperparameters"]["total_verification_timeout_s"]
        assert budgets["max_tuning_rounds"] == legacy["hyperparameters"]["max_tuning_iterations"]
        assert runtime["environment"]["NCCL_BUFFSIZE"] == str(2 * legacy["hyperparameters"]["slice_size_bytes"])
    for legacy_path in sorted((legacy_root / "topo").glob("*.json")):
        legacy = json.loads(legacy_path.read_text())
        topology = json.loads((FIXTURES / "topology" / legacy_path.name).read_text())
        assert topology["ranks"] == legacy["ranks"]
        assert [n["ranks"] for n in topology["nodes"]] == [n["ranks"] for n in legacy["nodes"]]
        legacy_links = {(e["src"], e["dst"]): e for e in legacy["directed_links"]}
        links = {(e["src"], e["dst"]): e for e in topology["directed_links"]}
        assert links.keys() == legacy_links.keys()
        reference = min(e["beta"] for e in legacy_links.values()) / min(e["beta_us_per_byte"] for e in links.values())
        for key, edge in links.items():
            assert edge["alpha_us"] == legacy_links[key]["alpha"]
            assert abs(edge["beta_us_per_byte"] * reference - legacy_links[key]["beta"]) < 1e-9
            assert edge["resources"] == legacy_links[key]["resources"]
        assert {r["id"] for r in topology["shared_resources"]} == {r["id"] for r in legacy["shared_resources"]}
