from pathlib import Path

import pytest

from vericcl.provenance import ALLOWED_TACCL_REFERENCES


pytestmark = pytest.mark.phase07


PROJECT_ROOT = Path(__file__).parents[2]
TOKEN = "ta" + "ccl"
SOURCE_SUFFIXES = {".py", ".json", ".xml"}


def _observed_references():
    observed = {}
    roots = (PROJECT_ROOT / "vericcl", PROJECT_ROOT / "tests")
    files = [PROJECT_ROOT / "setup.py"]
    for root in roots:
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix in SOURCE_SUFFIXES
        )
    for path in files:
        lines = tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if TOKEN in line
        )
        if lines:
            observed[str(path.relative_to(PROJECT_ROOT))] = lines
    return observed


def test_retained_legacy_references_match_explicit_allowlist():
    observed = _observed_references()

    assert set(observed) == set(ALLOWED_TACCL_REFERENCES)
    assert all(len(lines) == 1 for lines in observed.values())
    assert all(
        isinstance(reason, str)
        and reason
        and reason.isascii()
        for reason in ALLOWED_TACCL_REFERENCES.values()
    )


def test_legacy_examples_and_templates_are_packaged_under_vericcl():
    examples = PROJECT_ROOT / "vericcl" / "examples"

    assert (examples / "legacy" / "topo" / "topo-ndv2-1MB.json").is_file()
    assert (examples / "legacy" / "sketch" / "sk2-ndv2-n2.json").is_file()
    assert (
        examples / "legacy" / "Allgather.n16-1MB_i8_v1.xml"
    ).is_file()
    assert (
        examples / "templates" / "mesh16" / "Allreduce_template.json"
    ).is_file()
