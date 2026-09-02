from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Mapping, Optional
import uuid

from vericcl.errors import SemanticError
from vericcl.verification.online.runner import ProcessRequest, ProcessResult


REMOTE_ENVIRONMENT_NAMES = frozenset(
    {
        "PATH",
        "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
    }
)
REMOTE_ENVIRONMENT_PREFIXES = ("NCCL_", "MSCCL_", "VERICCL_")

_HOST_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_REMOTE_PATH_PATTERN = re.compile(r"^[A-Za-z0-9_./-]+$")
_REMOTE_ARGUMENT_PATTERN = re.compile(r"^[A-Za-z0-9_./,:=+@%-]*$")


def _remote_host(value: object) -> str:
    if not isinstance(value, str) or not _HOST_PATTERN.fullmatch(value):
        raise SemanticError("remote host is invalid")
    return value


def _executor(value: object):
    if not callable(getattr(value, "run", None)):
        raise SemanticError("remote delegate must provide run(request)")
    return value


def _remote_path(value: Path) -> str:
    text = str(value)
    if not _REMOTE_PATH_PATTERN.fullmatch(text):
        raise SemanticError("remote path contains unsafe characters")
    return text


def _remote_argument(value: str) -> str:
    if not _REMOTE_ARGUMENT_PATTERN.fullmatch(value):
        raise SemanticError("remote argument contains unsafe characters")
    return value


def _checked_result(result: object, label: str) -> ProcessResult:
    if not isinstance(result, ProcessResult):
        raise SemanticError("{} returned an invalid process result".format(label))
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise SemanticError(
            "{} failed with status {}: {}".format(
                label,
                result.returncode,
                detail,
            )
        )
    return result


@dataclass(frozen=True)
class ExperimentPathPolicy:
    root: Path

    def __post_init__(self) -> None:
        candidate = Path(self.root)
        if not candidate.is_absolute():
            raise SemanticError("experiment root must be absolute")
        object.__setattr__(self, "root", candidate.resolve())

    def require_allowed(self, value: Path) -> Path:
        candidate = Path(value)
        if not candidate.is_absolute():
            raise SemanticError("experiment path must be absolute")
        path = candidate.resolve()
        if not path.is_relative_to(self.root):
            raise SemanticError("path is outside the experiment root")
        return path


class SshFileStager:
    def __init__(
        self,
        *,
        delegate,
        remote_host: str,
        path_policy: ExperimentPathPolicy,
        local_environment: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._delegate = _executor(delegate)
        self._remote_host = _remote_host(remote_host)
        if not isinstance(path_policy, ExperimentPathPolicy):
            raise SemanticError("remote path policy is invalid")
        self._path_policy = path_policy
        self._local_environment = dict(
            os.environ if local_environment is None else local_environment
        )

    @property
    def remote_host(self) -> str:
        return self._remote_host

    @property
    def path_policy(self) -> ExperimentPathPolicy:
        return self._path_policy

    def _run(self, command, label: str) -> ProcessResult:
        request = ProcessRequest(
            command=tuple(command),
            environment=self._local_environment,
            label=label,
        )
        return _checked_result(self._delegate.run(request), label)

    def ensure_directory(self, path: Path) -> None:
        remote = self._path_policy.require_allowed(path)
        self._run(
            (
                "ssh",
                "-o",
                "BatchMode=yes",
                self._remote_host,
                "mkdir",
                "-p",
                _remote_path(remote),
            ),
            "remote directory creation",
        )

    def upload(self, local: Path, remote: Path) -> None:
        source = self._path_policy.require_allowed(local)
        destination = self._path_policy.require_allowed(remote)
        if not source.is_file() or source.stat().st_size <= 0:
            raise SemanticError("upload source must be a non-empty regular file")
        temporary = destination.with_name(
            ".{}.tmp-{}".format(destination.name, uuid.uuid4().hex)
        )
        self.ensure_directory(destination.parent)
        self._run(
            (
                "scp",
                "-q",
                str(source),
                "{}:{}".format(
                    self._remote_host,
                    _remote_path(temporary),
                ),
            ),
            "remote file upload",
        )
        self._run(
            (
                "ssh",
                "-o",
                "BatchMode=yes",
                self._remote_host,
                "mv",
                "-f",
                _remote_path(temporary),
                _remote_path(destination),
            ),
            "remote file activation",
        )

    def fetch(self, remote: Path, local: Path) -> None:
        source = self._path_policy.require_allowed(remote)
        destination = self._path_policy.require_allowed(local)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            ".{}.tmp-{}".format(destination.name, uuid.uuid4().hex)
        )
        try:
            self._run(
                (
                    "scp",
                    "-q",
                    "{}:{}".format(
                        self._remote_host,
                        _remote_path(source),
                    ),
                    str(temporary),
                ),
                "remote file download",
            )
            if not temporary.is_file() or temporary.stat().st_size <= 0:
                raise SemanticError(
                    "downloaded file must be a non-empty regular file"
                )
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()


class SshStagingCommandExecutor:
    def __init__(
        self,
        *,
        delegate,
        stager,
        remote_host: str,
        path_policy: ExperimentPathPolicy,
        local_environment: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._delegate = _executor(delegate)
        if not callable(getattr(stager, "upload", None)) or not callable(
            getattr(stager, "ensure_directory", None)
        ):
            raise SemanticError("remote stager is invalid")
        self._stager = stager
        self._remote_host = _remote_host(remote_host)
        if not isinstance(path_policy, ExperimentPathPolicy):
            raise SemanticError("remote path policy is invalid")
        self._path_policy = path_policy
        self._local_environment = dict(
            os.environ if local_environment is None else local_environment
        )

    @staticmethod
    def _allowed_environment(environment: Mapping[str, str]):
        result = []
        for key, value in sorted(environment.items()):
            if key not in REMOTE_ENVIRONMENT_NAMES and not key.startswith(
                REMOTE_ENVIRONMENT_PREFIXES
            ):
                continue
            result.append((_remote_argument(key), _remote_argument(value)))
        return tuple(result)

    def run(self, request: ProcessRequest) -> ProcessResult:
        if not isinstance(request, ProcessRequest):
            raise SemanticError("remote executor requires a ProcessRequest")
        if request.cwd is not None:
            raise SemanticError("remote executor does not support a process cwd")
        command = tuple(_remote_argument(value) for value in request.command)

        xml_value = request.environment.get("MSCCL_XML_FILES")
        if xml_value:
            xml_path = self._path_policy.require_allowed(Path(xml_value))
            self._stager.upload(xml_path, xml_path)

        trace_value = request.environment.get("VERICCL_TRACE_FILE_PREFIX")
        if trace_value:
            trace_prefix = self._path_policy.require_allowed(Path(trace_value))
            self._stager.ensure_directory(trace_prefix.parent)

        assignments = tuple(
            "{}={}".format(key, value)
            for key, value in self._allowed_environment(request.environment)
        )
        remote_request = ProcessRequest(
            command=(
                "ssh",
                "-o",
                "BatchMode=yes",
                self._remote_host,
                "env",
            )
            + assignments
            + command,
            environment=self._local_environment,
            label="remote {}".format(request.label),
            timeout_s=request.timeout_s,
        )
        result = self._delegate.run(remote_request)
        if not isinstance(result, ProcessResult):
            raise SemanticError("remote delegate returned an invalid process result")
        return result
