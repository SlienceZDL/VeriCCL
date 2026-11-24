from argparse import Namespace
from dataclasses import replace
import json

import pytest


@pytest.mark.parametrize('mode', ['clock'])
def test_experiment_entry_preserves_explicit_hostfiles(tmp_path, monkeypatch, mode):
    import vericcl.experiments.v100 as v100
    import vericcl.experiments.overhead_jobs as jobs
    from tests.unit.experiments.test_v100 import _config
    config = replace(_config(tmp_path), remote_host='compute-a')
    for ranks in (8, 16):
        v100.write_hostfile(config.hostfile(ranks), ranks, hostnames=('compute-a', 'compute-b'))
    originals = {ranks: config.hostfile(ranks).read_bytes() for ranks in (8, 16)}
    monkeypatch.setattr(v100, 'load_v100_config', lambda path: config)
    class ReachedDependencies(Exception):
        pass
    def dependencies(actual, ranks):
        assert actual == config
        assert {count: config.hostfile(count).read_bytes() for count in (8, 16)} == originals
        raise ReachedDependencies
    monkeypatch.setattr(v100, '_runtime_dependencies', dependencies)
    args = Namespace(config='unused.json', output=str(config.experiment_root / 'result'), ranks=8, mode=mode)
    with pytest.raises(ReachedDependencies):
        jobs.run_clock(args)


def test_bound_runtime_follows_explicit_hostfiles(tmp_path):
    import vericcl.experiments.v100 as v100
    from tests.unit.experiments.test_v100 import REPO_ROOT, CASE_FILES, _config
    config = replace(_config(tmp_path), remote_host='compute-a')
    for ranks in (8, 16):
        v100.write_hostfile(config.hostfile(ranks), ranks, hostnames=('compute-a', 'compute-b'))
    originals = {ranks: config.hostfile(ranks).read_bytes() for ranks in (8, 16)}
    runtime = json.loads((REPO_ROOT / CASE_FILES['runtime']).read_text())

    bound = v100.bind_runtime(config, runtime)

    assert [entry['host'] for entry in bound['rank_map']] == ['compute-a'] * 4 + ['compute-b'] * 4
    assert {count: config.hostfile(count).read_bytes() for count in (8, 16)} == originals
