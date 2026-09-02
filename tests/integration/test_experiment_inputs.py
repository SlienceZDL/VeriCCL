from pathlib import Path

from vericcl.input.loader import resolve_inputs
from vericcl.planner import build_plan
from vericcl.topology.loader import load_topology


def test_all_experiment_inputs_resolve_and_build():
    root = Path(__file__).parents[2]
    atom = root / "vericcl/examples/atom/default.json"
    cases = []
    for topology_path in sorted((root / "exp/topo").glob("*.json")):
        sketch_root = root / "exp/sketch" / topology_path.stem
        for sketch in sorted(sketch_root.glob("*/*.json")):
            resolved = resolve_inputs(topology_path, sketch, atom)
            assert resolved.solver.max_channels == 16
            assert resolved.hyperparameters.max_calibration_channels == 16
            build_plan(resolved, load_topology(resolved))
            cases.append((topology_path, sketch))
    assert len(cases) == 36
