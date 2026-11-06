"""Toolchain failures still produce a complete, inspectable build result."""
import importlib
import json
from pathlib import Path
import sys
import zipfile

import pytest


def toolchain(tmp_path, failure):
    cuda, mpi, runtime = (tmp_path / name for name in ('cuda', 'mpi', 'msccl'))
    (cuda / 'bin').mkdir(parents=True)
    mpi.mkdir()
    (runtime / 'lib').mkdir(parents=True)
    (runtime / 'lib/libnccl.so').touch()
    nvcc = cuda / 'bin/nvcc'
    nvcc.write_text('#!' + sys.executable + '\nraise AssertionError("MPI failure must stop before compilation")\n')
    nvcc.chmod(0o755)
    wrapper = mpi / 'mpicxx'
    wrapper.write_text('#!' + sys.executable + '\nimport sys, time\n'
        'print("partial MPI probe output", flush=True)\n'
        'print("MPI probe diagnostic", file=sys.stderr, flush=True)\n' +
        ('time.sleep(30)\n' if failure == 'timeout' else 'raise SystemExit(17)\n'))
    wrapper.chmod(0o755)
    return ['--cuda-home', str(cuda), '--mpi-bin', str(mpi), '--msccl-build', str(runtime),
            '--arch', '70', '--timeout-s', '30']


@pytest.mark.parametrize('name', ['build_native', 'build_probes'])
@pytest.mark.parametrize('failure', ['failed', 'timeout'])
def test_mpi_probe_failure_is_finalized_with_durable_partial_output(tmp_path, name, failure):
    builder = importlib.import_module('runtime.' + name)
    output = tmp_path / 'output'
    arguments = toolchain(tmp_path, failure) + ['--output-dir', str(output)]
    assert builder.main(arguments) == 8
    flags = json.loads((output / 'process/mpi-flags.json').read_text())['compile']
    assert flags['status'] == failure
    assert flags['returncode'] == (17 if failure == 'failed' else None)
    assert 'partial MPI probe output' in flags['stdout']
    assert 'MPI probe diagnostic' in flags['stderr']
    assert flags['argv'][-1] == '--showme:compile'
    result = json.loads((output / 'process/result.json').read_text())
    assert result['compiled'] is False and result['error']
    metadata = json.loads((output / 'run.json').read_text())
    assert metadata['result']['contract_met'] is False
    assert metadata['result']['execution'] == 'failed'
    assert metadata['result']['exit_code'] == 8
    assert (output / 'report.md').is_file()
    with zipfile.ZipFile(output / 'details.zip') as archive:
        assert archive.testzip() is None
        assert any(path.endswith('process/mpi-flags.json') for path in archive.namelist())


@pytest.mark.parametrize('name', ['build_native', 'build_probes'])
def test_missing_toolchain_remains_a_precontext_input_error(tmp_path, name):
    builder = importlib.import_module('runtime.' + name)
    output = tmp_path / 'output'
    with pytest.raises(SystemExit) as error:
        builder.main(['--output-dir', str(output), '--cuda-home', str(tmp_path / 'missing'),
            '--mpi-bin', str(tmp_path / 'missing'), '--msccl-build', str(tmp_path / 'missing'),
            '--arch', '70', '--timeout-s', '30'])
    assert error.value.code == 2
    assert not output.exists()
