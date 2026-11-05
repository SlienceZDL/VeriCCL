import json
from pathlib import Path
import runpy


def test_package_uses_explicit_files_and_excludes_local_reports(tmp_path, monkeypatch):
    import setuptools
    root = Path(__file__).resolve().parents[2]
    private = ('requirements-lock/private-environment.json', 'tools/reports/delivery.json',
               'tests/results/run-summary.json', 'runtime/output/result.json',
               'examples/quickstart/input/private-environment.json')
    for relative in private:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    (tmp_path / 'README.md').write_text('public')
    (tmp_path / 'public-files.json').write_text(json.dumps(['README.md']))
    setup = tmp_path / 'setup.py'; setup.write_text((root / 'setup.py').read_text())
    captured = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setuptools, 'setup', lambda **values: captured.update(values))
    runpy.run_path(str(setup))
    shipped = {path for _, files in captured['data_files'] for path in files}
    assert shipped == {'README.md'}
    assert not shipped.intersection(private)
    distribution = captured['cmdclass']['sdist'](setuptools.Distribution())
    assert distribution.owner == distribution.group == 'root'


def test_source_manifest_lists_exact_public_paths():
    root = Path(__file__).resolve().parents[2]
    paths = json.loads((root / 'public-files.json').read_text())
    assert len(paths) == len(set(paths))
    assert 'README.md' in paths and 'runtime/build_runtime.py' in paths
    entries = (root / 'MANIFEST.in').read_text().splitlines()
    assert entries == ['include ' + path for path in paths]
    for name in paths:
        path = Path(name)
        assert not path.is_absolute() and '..' not in path.parts
        assert (root / path).is_file()
        assert not set(path.parts).intersection({'output', '.deps', '__pycache__', '.git'})
