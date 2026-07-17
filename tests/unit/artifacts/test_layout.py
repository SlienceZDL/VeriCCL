from dataclasses import replace

import pytest

from vericcl.artifacts.layout import create_run_layout
from vericcl.errors import SemanticError
from vericcl.semantics.collective import CollectiveKind

from tests.unit.xml.helpers import resolved


pytestmark = pytest.mark.phase07


def _resolved_allreduce():
    inputs = resolved(CollectiveKind.ALL_REDUCE)
    return replace(
        inputs,
        hyperparameters=replace(
            inputs.hyperparameters,
            total_size_bytes=256 * 1024 * 1024,
            slice_size_bytes=1024 * 1024,
        ),
    )


def test_run_layout_matches_spec(tmp_path):
    layout = create_run_layout(
        tmp_path,
        _resolved_allreduce(),
        run_id="0001",
    )

    assert layout.root.name == "vericcl_allreduce_256MiB_0001"
    assert layout.resolved_input.name == "resolved-input.json"
    assert layout.summary.name == "run-summary.json"
    assert layout.schedules.name == "schedules"
    assert layout.reports.name == "reports"
    assert layout.traces.name == "traces"
    assert layout.schedules.is_dir()
    assert layout.reports.is_dir()
    assert layout.traces.is_dir()


def test_layout_rejects_nonempty_existing_run_directory(tmp_path):
    root = tmp_path / "vericcl_allreduce_256MiB_0001"
    root.mkdir()
    (root / "user-data.txt").write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty"):
        create_run_layout(tmp_path, _resolved_allreduce(), run_id="0001")

    assert (root / "user-data.txt").read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("run_id", ("", "../escape", "a/b", "optimal"))
def test_layout_rejects_unsafe_or_misleading_run_ids(tmp_path, run_id):
    with pytest.raises(SemanticError, match="run_id"):
        create_run_layout(tmp_path, _resolved_allreduce(), run_id=run_id)
