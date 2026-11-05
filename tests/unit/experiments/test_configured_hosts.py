from argparse import Namespace
from dataclasses import replace
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('mode', ['clock', 'online'])
def test_experiment_entry_preserves_explicit_hostfiles(tmp_path, monkeypatch, mode):
    import vericcl.experiments.v100 as v100
    import vericcl.experiments.overhead_jobs as jobs
    import vericcl.verification.pipeline as pipeline
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
    if mode == 'online':
        monkeypatch.setattr(jobs, 'online_output_root', lambda args: config.experiment_root / 'result')
        monkeypatch.setattr(jobs, 'load_case', lambda args: (object(), object(), SimpleNamespace(rank_count=8)))
        monkeypatch.setattr(pipeline, 'validate_and_lower_candidate', lambda *args: SimpleNamespace(artifact=SimpleNamespace(xml_text='<algo/>')))
        monkeypatch.setattr(jobs, 'online_eligible', lambda outcome: True)
    with pytest.raises(ReachedDependencies):
        (jobs.run_clock if mode == 'clock' else jobs.run_online_once)(args)
