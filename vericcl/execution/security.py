"""Keep credentials out of evidence while preserving reproducible settings."""
from __future__ import annotations

import json
import os
import pwd
from pathlib import Path
import re
from collections.abc import Mapping


ENVIRONMENT_ALLOWLIST = frozenset({
    "PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "CUDA_HOME", "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES", "NCCL_ALGO", "NCCL_PROTO", "NCCL_BUFFSIZE",
    "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS", "NCCL_IB_HCA", "NCCL_IB_GID_INDEX",
    "NCCL_SOCKET_IFNAME", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE",
    "MSCCL_XML_FILES", "NCCL_TESTS_ROOT", "MSCCL_ROOT", "OMP_NUM_THREADS",
    "OMPI_MCA_btl", "OMPI_MCA_pml", "UCX_NET_DEVICES", "UCX_TLS",
    "OMPI_MCA_btl_tcp_if_include", "OMPI_MCA_oob_tcp_if_include",
})
_SECRET_KEY = re.compile(r"(?:^|[_-])(password|passwd|token|secret|private[_-]?key|api[_-]?key|credential)(?:$|[_-])", re.I)
_ASSIGNMENT = re.compile(r"(?i)((?:password|passwd|token|secret|api[_-]?key)\s*[:=]\s*)([^\s,;]+)")


def sensitive_key(key: object) -> bool:
    return isinstance(key, str) and bool(_SECRET_KEY.search(key.lstrip("-")))


class Redactor:
    def __init__(self, secrets=(), environment=None):
        environment = environment or {}
        values = list(secrets) + [v for k, v in environment.items() if sensitive_key(k)]
        self.secrets = tuple(sorted({v for v in values if isinstance(v, str) and v}, key=len, reverse=True))

    def text(self, text: str) -> str:
        for secret in self.secrets:
            text = text.replace(secret, "[REDACTED]")
        return _ASSIGNMENT.sub(r"\1[REDACTED]", text)

    def argv(self, argv) -> list[str]:
        result, hide_next = [], False
        for value in argv:
            if hide_next:
                result.append("[REDACTED]")
                hide_next = False
            elif "=" in value and sensitive_key(value.split("=", 1)[0]):
                result.append(value.split("=", 1)[0] + "=[REDACTED]")
            elif sensitive_key(value):
                result.append(value)
                hide_next = True
            else:
                result.append(self.text(value))
        return result

    def value(self, value):
        if isinstance(value, Mapping):
            result = {}
            for key, item in value.items():
                if sensitive_key(key):
                    result[key] = "[REDACTED]"
                elif key in {"environment", "env"} and isinstance(item, Mapping):
                    result[key] = {k: self.text(v) for k, v in item.items() if k in ENVIRONMENT_ALLOWLIST}
                elif key in {"argv", "command"} and isinstance(item, (tuple, list)):
                    result[key] = self.argv(item)
                else:
                    result[key] = self.value(item)
            return result
        if isinstance(value, (list, tuple)):
            return [self.value(item) for item in value]
        if isinstance(value, str):
            return self.text(value)
        return value

    def check_snapshot(self, data: bytes) -> None:
        text = data.decode("utf-8")
        if any(secret in text for secret in self.secrets):
            raise ValueError("credential values are not allowed in input snapshots")
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            if _ASSIGNMENT.search(text):
                raise ValueError("credential values are not allowed in input snapshots") from None
            return
        def contains(item):
            if isinstance(item, dict):
                return any(sensitive_key(k) or contains(v) for k, v in item.items())
            return isinstance(item, list) and any(contains(v) for v in item)
        if contains(value):
            raise ValueError("credential fields are not allowed in input snapshots")


class StreamRedactor:
    """Redact known secret values even when reads split a value in two."""
    def __init__(self, secrets):
        self.secrets = tuple(s.encode("utf-8") for s in secrets)
        self.pending = b""

    def feed(self, data: bytes, *, final=False) -> bytes:
        self.pending += data
        output = bytearray()
        while self.pending:
            match = next((s for s in self.secrets if self.pending.startswith(s)), None)
            if match is not None:
                output.extend(b"[REDACTED]")
                self.pending = self.pending[len(match):]
            elif not final and any(s.startswith(self.pending) for s in self.secrets):
                break
            else:
                # Copy up to the next potentially matching first byte.
                positions = [self.pending.find(s[:1], 1) for s in self.secrets]
                count = min((p for p in positions if p >= 0), default=len(self.pending))
                output.extend(self.pending[:count])
                self.pending = self.pending[count:]
        return bytes(output)


def _account_home_paths() -> tuple[Path, Path]:
    home = Path(pwd.getpwuid(os.geteuid()).pw_dir)
    resolved = home.resolve()
    if not home.is_absolute() or resolved == Path('/'):
        raise ValueError('executing account home must be an absolute non-root directory')
    return home, resolved


def executing_user_home() -> Path:
    """Resolve the effective UID account home independently of shell variables."""
    return _account_home_paths()[1]


def is_foreign_home_path(path) -> bool:
    """Identify dependency paths owned by a different home-directory account."""
    home, resolved_home = _account_home_paths()
    parents = {home.parent, resolved_home.parent, Path('/home'), Path('/Users')}
    for item in (Path(path), Path(path).resolve()):
        for parent in parents:
            if parent == Path('/'):
                continue
            if item.is_relative_to(parent) and not (
                    item.is_relative_to(home) or item.is_relative_to(resolved_home)):
                return True
    return False
