from pathlib import Path

import pytest

from vericcl.errors import SemanticError
from vericcl.experiments.remote import (
    ExperimentPathPolicy,
    RemoteTraceCollector,
    SshFileStager,
    SshStagingCommandExecutor,
)
from vericcl.verification.online.runner import (
    ProcessRequest,
    ProcessResult,
    TraceCollectionRequest,
    TraceCollectionResult,
)
from vericcl.verification.online.trace_analysis import TraceAnalysis
from vericcl.xml.endpoints import EndpointType
from vericcl.xml.trace_sidecar import TraceSidecar, TraceStepMetadata


class RecordingExecutor:
    def __init__(self, results=(), download_content=None):
        self.calls = []
        self.results = list(results)
        self.download_content = download_content

    def run(self, request):
        self.calls.append(request)
        if (
            self.download_content is not None
            and request.command[0] == "scp"
            and ":" in request.command[-2]
        ):
            Path(request.command[-1]).write_bytes(self.download_content)
        if self.results:
            return self.results.pop(0)
        return ProcessResult(0, "", "")


class RecordingStager:
    def __init__(self, *, create_downloads=False, fail_fetch=False):
        self.uploads = []
        self.directories = []
        self.downloads = []
        self.create_downloads = create_downloads
        self.fail_fetch = fail_fetch

    def upload(self, local, remote):
        self.uploads.append((Path(local), Path(remote)))

    def ensure_directory(self, path):
        self.directories.append(Path(path))

    def fetch(self, remote, local):
        remote_path = Path(remote)
        local_path = Path(local)
        self.downloads.append((remote_path, local_path))
        if self.fail_fetch:
            raise SemanticError("download failed")
        if self.create_downloads:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(b"trace")


def _request(xml, trace_prefix, **environment):
    values = {
        "PATH": "/usr/bin:/bin",
        "LD_LIBRARY_PATH": "/opt/msccl/lib",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "MSCCL_XML_FILES": str(xml),
        "VERICCL_TRACE_FILE_PREFIX": str(trace_prefix),
        "NCCL_PROTO": "Simple",
        "GRB_LICENSE_FILE": "/secret/gurobi.lic",
        "SSH_AUTH_SOCK": "/secret/agent.sock",
        "UNRELATED_SECRET": "do-not-forward",
    }
    values.update(environment)
    return ProcessRequest(
        command=("/opt/openmpi/bin/mpirun", "-np", "2", "/opt/test"),
        environment=values,
        label="remote test",
        timeout_s=30.0,
    )


def _trace_request(tmp_path, *, rank_count):
    entry = TraceStepMetadata(
        rank=0,
        tb_id=0,
        step_index=0,
        xml_step_index=0,
        step_id="copy:cpy",
        transfer_id="copy",
        endpoint_type=EndpointType.COPY,
        runtime_endpoint_type=6,
        peer=-1,
        runtime_channel=0,
        stage_id=-1,
        atom_ids=(),
        flow_ids=(),
        lane=None,
        semantic_predecessor_ids=(),
        member_slice_ids=frozenset(),
    )
    sidecar = TraceSidecar(
        xml_sha256="a" * 64,
        schedule_id="remote-test",
        rank_count=rank_count,
        entries={entry.key: entry},
    )
    return TraceCollectionRequest(
        sidecar=sidecar,
        file_prefix=tmp_path / "trace" / "step",
        rank_count=rank_count,
        clock_sync_output="clock-sync",
        max_clock_uncertainty_us=10.0,
    )


def _trace_result():
    return TraceCollectionResult(
        TraceAnalysis((), (), (), (), True),
        (Path("/tmp/trace.rank-0.bin"),),
        True,
    )


def test_path_policy_rejects_path_outside_experiment_root(tmp_path):
    policy = ExperimentPathPolicy(tmp_path / "allowed")

    with pytest.raises(SemanticError, match="experiment root"):
        policy.require_allowed(tmp_path / "outside.xml")
    with pytest.raises(SemanticError, match="absolute"):
        policy.require_allowed(Path("relative.xml"))


def test_remote_executor_stages_xml_and_executes_on_node4(tmp_path):
    delegate = RecordingExecutor()
    stager = RecordingStager()
    executor = SshStagingCommandExecutor(
        delegate=delegate,
        stager=stager,
        remote_host="10.0.0.104",
        path_policy=ExperimentPathPolicy(tmp_path),
    )
    xml = tmp_path / "case.xml"
    xml.write_text("<algo/>", encoding="ascii")
    trace_prefix = tmp_path / "trace" / "step"

    result = executor.run(_request(xml, trace_prefix))

    assert stager.uploads == [(xml, xml)]
    assert stager.directories == [trace_prefix.parent]
    assert result.returncode == 0
    remote = delegate.calls[-1]
    assert remote.command[:4] == (
        "ssh",
        "-o",
        "BatchMode=yes",
        "10.0.0.104",
    )
    assert remote.command[4] == "env"
    joined = "\n".join(remote.command)
    assert "NCCL_PROTO=Simple" in remote.command
    assert "GRB_LICENSE_FILE" not in joined
    assert "SSH_AUTH_SOCK" not in joined
    assert "UNRELATED_SECRET" not in joined


def test_remote_executor_returns_remote_nonzero_result_unchanged(tmp_path):
    expected = ProcessResult(7, "remote stdout", "remote stderr")
    delegate = RecordingExecutor((expected,))
    stager = RecordingStager()
    executor = SshStagingCommandExecutor(
        delegate=delegate,
        stager=stager,
        remote_host="10.0.0.104",
        path_policy=ExperimentPathPolicy(tmp_path),
    )
    xml = tmp_path / "case.xml"
    xml.write_text("<algo/>", encoding="ascii")

    assert executor.run(_request(xml, tmp_path / "trace" / "step")) is expected


@pytest.mark.parametrize(
    "unsafe_value",
    ("INFO\nINJECTED=1", "INFO;touch-/tmp/injected"),
)
def test_remote_executor_rejects_unsafe_runtime_paths_and_values(
    tmp_path,
    unsafe_value,
):
    delegate = RecordingExecutor()
    executor = SshStagingCommandExecutor(
        delegate=delegate,
        stager=RecordingStager(),
        remote_host="10.0.0.104",
        path_policy=ExperimentPathPolicy(tmp_path / "allowed"),
    )
    outside = tmp_path / "outside.xml"
    outside.write_text("<algo/>", encoding="ascii")

    with pytest.raises(SemanticError, match="experiment root"):
        executor.run(_request(outside, tmp_path / "allowed/trace/step"))

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    xml = allowed / "case.xml"
    xml.write_text("<algo/>", encoding="ascii")
    with pytest.raises(SemanticError, match="unsafe"):
        executor.run(
            _request(
                xml,
                allowed / "trace" / "step",
                NCCL_DEBUG=unsafe_value,
            )
        )


def test_file_stager_uploads_via_temporary_remote_sibling(tmp_path):
    root = tmp_path / "experiment"
    root.mkdir()
    source = root / "case.xml"
    source.write_text("<algo/>", encoding="ascii")
    delegate = RecordingExecutor()
    stager = SshFileStager(
        delegate=delegate,
        remote_host="10.0.0.104",
        path_policy=ExperimentPathPolicy(root),
    )

    stager.upload(source, source)

    commands = tuple(call.command for call in delegate.calls)
    assert commands[0] == (
        "ssh", "-o", "BatchMode=yes", "10.0.0.104",
        "mkdir", "-p", str(root),
    )
    assert commands[1][:2] == ("scp", "-q")
    assert commands[1][2] == str(source)
    assert commands[1][3].startswith("10.0.0.104:{}".format(root))
    assert commands[2][:6] == (
        "ssh", "-o", "BatchMode=yes", "10.0.0.104",
        "mv", "-f",
    )
    assert commands[2][-1] == str(source)


def test_file_stager_stops_after_failed_upload(tmp_path):
    root = tmp_path / "experiment"
    root.mkdir()
    source = root / "case.xml"
    source.write_text("<algo/>", encoding="ascii")
    delegate = RecordingExecutor(
        (
            ProcessResult(0, "", ""),
            ProcessResult(1, "", "upload failed"),
        )
    )
    stager = SshFileStager(
        delegate=delegate,
        remote_host="10.0.0.104",
        path_policy=ExperimentPathPolicy(root),
    )

    with pytest.raises(SemanticError, match="upload failed"):
        stager.upload(source, source)
    assert len(delegate.calls) == 2


def test_file_stager_fetches_nonempty_file_atomically(tmp_path):
    root = tmp_path / "experiment"
    root.mkdir()
    remote = root / "remote" / "step.rank-0.bin"
    local = root / "local" / "step.rank-0.bin"
    delegate = RecordingExecutor(download_content=b"trace")
    stager = SshFileStager(
        delegate=delegate,
        remote_host="10.0.0.104",
        path_policy=ExperimentPathPolicy(root),
    )

    stager.fetch(remote, local)

    assert local.read_bytes() == b"trace"
    assert tuple(call.command[0] for call in delegate.calls) == ("scp",)


def test_file_stager_rejects_empty_download(tmp_path):
    root = tmp_path / "experiment"
    root.mkdir()
    remote = root / "remote" / "step.rank-0.bin"
    local = root / "local" / "step.rank-0.bin"
    stager = SshFileStager(
        delegate=RecordingExecutor(download_content=b""),
        remote_host="10.0.0.104",
        path_policy=ExperimentPathPolicy(root),
    )

    with pytest.raises(SemanticError, match="non-empty"):
        stager.fetch(remote, local)
    assert not local.exists()


@pytest.mark.parametrize("host", ("", "-unsafe", "node4\ncommand"))
def test_remote_components_reject_unsafe_host(tmp_path, host):
    with pytest.raises(SemanticError, match="host"):
        SshFileStager(
            delegate=RecordingExecutor(),
            remote_host=host,
            path_policy=ExperimentPathPolicy(tmp_path),
        )


def test_remote_trace_collector_fetches_node4_ranks_before_analysis(tmp_path):
    stager = RecordingStager(create_downloads=True)
    delegated = []
    collector = RemoteTraceCollector(
        stager=stager,
        delegate=lambda request: delegated.append(request) or _trace_result(),
    )
    request = _trace_request(tmp_path, rank_count=8)

    result = collector(request)

    assert tuple(path.name for _, path in stager.downloads) == (
        "step.rank-0.bin",
        "step.rank-1.bin",
        "step.rank-2.bin",
        "step.rank-3.bin",
    )
    assert delegated == [request]
    assert result.complete is True


def test_remote_trace_collector_fetches_rank_zero_for_two_rank_calibration(
    tmp_path,
):
    stager = RecordingStager(create_downloads=True)
    collector = RemoteTraceCollector(
        stager=stager,
        delegate=lambda request: _trace_result(),
    )

    collector(_trace_request(tmp_path, rank_count=2))

    assert tuple(path.name for _, path in stager.downloads) == (
        "step.rank-0.bin",
    )


def test_remote_trace_collector_stops_when_fetch_fails(tmp_path):
    delegated = []
    collector = RemoteTraceCollector(
        stager=RecordingStager(fail_fetch=True),
        delegate=lambda request: delegated.append(request) or _trace_result(),
    )

    with pytest.raises(SemanticError, match="download failed"):
        collector(_trace_request(tmp_path, rank_count=8))
    assert delegated == []


def test_remote_trace_collector_rejects_odd_rank_count(tmp_path):
    collector = RemoteTraceCollector(
        stager=RecordingStager(),
        delegate=lambda request: _trace_result(),
    )

    with pytest.raises(SemanticError, match="even"):
        collector(_trace_request(tmp_path, rank_count=3))
