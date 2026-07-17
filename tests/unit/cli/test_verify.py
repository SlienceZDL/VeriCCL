from pathlib import Path
from types import SimpleNamespace

import pytest

from vericcl.cli.main import build_parser, main
from vericcl.cli.verify import infer_schedule_sidecar


pytestmark = pytest.mark.phase07


def _argv(*extra):
    return [
        "verify",
        "--topology",
        "topology.json",
        "--sketch",
        "sketch.json",
        "--atoms",
        "atom.json",
        "--xml",
        "schedule.xml",
        "--run-id",
        "verify-test",
        *extra,
    ]


def _result(*, final=True, online="not_run", runtime=True):
    candidate = SimpleNamespace(
        candidate_id="candidate-0",
        runtime_compatible=runtime,
        validation={"online": online},
    )
    return SimpleNamespace(
        status="valid" if final else "invalid",
        message="verification complete",
        final_candidate_id="candidate-0" if final else None,
        final_xml=Path("/tmp/final.xml") if final else None,
        layout=SimpleNamespace(root=Path("/tmp/run")),
        candidates=(candidate,),
    )


def test_verify_parser_exposes_online_tune_timeout_and_sidecar():
    args = build_parser().parse_args(
        _argv(
            "--sidecar",
            "schedule.schedule.json",
            "--online",
            "--tune",
            "--timeout-s",
            "10800",
        )
    )

    assert args.sidecar == Path("schedule.schedule.json")
    assert args.online is True
    assert args.tune is True
    assert args.timeout_s == 10800.0


@pytest.mark.parametrize(
    ("xml_name", "sidecar_name"),
    (
        ("schedule.xml", "schedule.schedule.json"),
        ("schedule.candidate.xml", "schedule.schedule.json"),
        (
            "vericcl_allreduce_2MiB_final.xml",
            "vericcl_allreduce_2MiB_final.schedule.json",
        ),
    ),
)
def test_sidecar_path_is_inferred_from_xml_name(xml_name, sidecar_name):
    assert infer_schedule_sidecar(Path("/tmp") / xml_name).name == sidecar_name


def test_invalid_schedule_returns_three(monkeypatch, capsys):
    monkeypatch.setattr(
        "vericcl.cli.verify.execute_verify",
        lambda context: _result(final=False),
    )
    monkeypatch.setattr("vericcl.cli.verify.require_input_files", lambda *args: None)

    assert main(_argv()) == 3
    captured = capsys.readouterr()
    assert "no semantic-valid candidate" in captured.err.lower()


def test_runtime_warning_with_candidate_output_returns_success(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        "vericcl.cli.verify.execute_verify",
        lambda context: _result(runtime=False),
    )
    monkeypatch.setattr("vericcl.cli.verify.require_input_files", lambda *args: None)

    assert main(_argv()) == 0
    captured = capsys.readouterr()
    assert "runtime_compatible=false" in captured.out
    assert captured.err == ""


def test_requested_online_failure_returns_four(monkeypatch, capsys):
    monkeypatch.setattr(
        "vericcl.cli.verify.execute_verify",
        lambda context: _result(online="failed"),
    )
    monkeypatch.setattr("vericcl.cli.verify.require_input_files", lambda *args: None)
    monkeypatch.setattr(
        "vericcl.cli.verify.build_online_context_factory",
        lambda: lambda *args: None,
    )

    assert main(_argv("--online")) == 4
    captured = capsys.readouterr()
    assert "online validation failed" in captured.err.lower()


def test_unexpected_internal_error_returns_five(monkeypatch, capsys):
    def fail(context):
        raise RuntimeError("forced internal failure")

    monkeypatch.setattr("vericcl.cli.verify.execute_verify", fail)
    monkeypatch.setattr("vericcl.cli.verify.require_input_files", lambda *args: None)

    assert main(_argv()) == 5
    captured = capsys.readouterr()
    assert "forced internal failure" in captured.err


@pytest.mark.parametrize(("extra", "expected"), (((), 3), (("--online",), 4)))
def test_timeout_uses_offline_or_online_exit_code(
    monkeypatch,
    capsys,
    extra,
    expected,
):
    def fail(context):
        raise TimeoutError("forced timeout")

    monkeypatch.setattr("vericcl.cli.verify.execute_verify", fail)
    monkeypatch.setattr("vericcl.cli.verify.require_input_files", lambda *args: None)
    if extra:
        monkeypatch.setattr(
            "vericcl.cli.verify.build_online_context_factory",
            lambda: lambda *args: None,
        )

    assert main(_argv(*extra)) == expected
    assert "forced timeout" in capsys.readouterr().err
