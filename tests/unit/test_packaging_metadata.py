from pathlib import Path
import runpy
from unittest import mock

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


PROJECT_ROOT = Path(__file__).parents[2]
SETUP_PATH = PROJECT_ROOT / "setup.py"


def _setup_metadata():
    with mock.patch("setuptools.setup") as setup:
        runpy.run_path(str(SETUP_PATH), run_name="__vericcl_setup_test__")
    return setup.call_args.kwargs


def _selected_dd(version):
    environment = default_environment()
    environment["python_version"] = version
    requirements = (
        Requirement(value)
        for value in _setup_metadata()["install_requires"]
        if Requirement(value).name == "dd"
    )
    return tuple(
        requirement
        for requirement in requirements
        if requirement.marker.evaluate(environment)
    )


def test_setup_declares_supported_python_range():
    assert _setup_metadata()["python_requires"] == ">=3.10,<3.14"


def test_setup_selects_one_dd_series_for_each_supported_python():
    expected = {
        "3.10": ">=0.5.7,<0.6",
        "3.11": ">=0.6,<0.7",
        "3.12": ">=0.6,<0.7",
        "3.13": ">=0.6,<0.7",
    }
    for version, specifier in expected.items():
        selected = _selected_dd(version)
        assert len(selected) == 1
        assert selected[0].specifier == SpecifierSet(specifier)
