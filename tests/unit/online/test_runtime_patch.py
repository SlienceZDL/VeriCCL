import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


pytestmark = pytest.mark.phase06


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = REPOSITORY_ROOT / "runtime" / "msccl-trace"
RUNTIME_README = RUNTIME_ROOT / "README.md"
METADATA_FILE = RUNTIME_ROOT / "upstream.json"
FORMAT_HEADER = RUNTIME_ROOT / "include" / "vericcl_trace_format.h"
PATCH_FILE = (
    RUNTIME_ROOT
    / "patches"
    / "0001-vericcl-fixed-step-trace.patch"
)
HOST_SIGNATURE_PATCH = (
    RUNTIME_ROOT
    / "patches"
    / "0002-vericcl-host-step-signature.patch"
)
VERIFY_PATCH = RUNTIME_ROOT / "tools" / "verify_patch.py"
REFERENCE_ROOT = os.environ.get("VERICCL_MSCCL_REFERENCE_ROOT")
STRATEGY_PATTERN = re.compile(
    r"<!-- vericcl-msccl-strategy: ([a-z-]+) -->\s*"
    r"```bash\s*\n(.*?)\n```",
    re.DOTALL,
)


def test_setup_installs_both_runtime_patches():
    text = Path("setup.py").read_text(encoding="ascii")

    assert "0001-vericcl-fixed-step-trace.patch" in text
    assert "0002-vericcl-host-step-signature.patch" in text


EXPECTED_RECORD_FIELDS = (
    ("uint32_t", "rank"),
    ("uint16_t", "tb_id"),
    ("uint16_t", "step_index"),
    ("uint16_t", "endpoint_type"),
    ("int16_t", "peer"),
    ("uint16_t", "channel"),
    ("uint32_t", "iteration"),
    ("uint64_t", "tb_reach"),
    ("uint64_t", "dependency_done"),
    ("uint64_t", "transfer_start"),
    ("uint64_t", "transfer_end"),
    ("uint32_t", "flags"),
    ("uint32_t", "reserved"),
)

EXPECTED_PATCHED_FILES = {
    "src/collectives/device/msccl_interpreter.h",
    "src/collectives/device/primitives.h",
    "src/collectives/device/prims_ll.h",
    "src/collectives/device/prims_ll128.h",
    "src/collectives/device/prims_simple.h",
    "src/include/msccl.h",
    "src/include/vericcl_trace_format.h",
    "src/init.cc",
}


def _record_fields(header_text):
    declarations = re.findall(
        r"typedef\s+struct\s*\{(?P<body>.*?)\}\s*(?P<name>\w+)\s*;",
        header_text,
        flags=re.DOTALL,
    )
    body = next(
        body
        for body, name in declarations
        if name == "VericclRawStepTraceRecord"
    )
    return tuple(
        (field_type, field_name)
        for field_type, field_name in re.findall(
            r"\b(u?int(?:16|32|64)_t)\s+(\w+)\s*;",
            body,
        )
    )


def test_raw_record_has_stable_exact_layout_contract():
    header = FORMAT_HEADER.read_text(encoding="utf-8")

    assert _record_fields(header) == EXPECTED_RECORD_FIELDS
    assert "VERICCL_TRACE_MAGIC" in header
    assert "VERICCL_TRACE_VERSION" in header
    assert "sizeof(VericclRawStepTraceRecord) == 64" in header
    assert "offsetof(VericclRawStepTraceRecord, tb_reach) == 24" in header


def test_runtime_metadata_pins_reproducible_sources():
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))

    assert metadata["schema_version"] == 1
    assert metadata["upstream_repository"] == "https://github.com/microsoft/msccl.git"
    assert metadata["upstream_commit"] == "b23e9cd5dd63f82ee1c5aae7e0a2042079be903a"
    assert metadata["fork_repository"] == "https://github.com/SlienceZDL/VeriCCL-MSCCL.git"
    assert metadata["fork_tag"] == "vericcl-runtime-v0.1.0"


def test_runtime_metadata_binds_published_commit_and_every_runtime_file():
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))

    assert re.fullmatch(r"[0-9a-f]{40}", metadata.get("patched_commit", ""))
    assert set(metadata.get("patched_files", {})) == EXPECTED_PATCHED_FILES
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", digest)
        for digest in metadata.get("patched_files", {}).values()
    )


def test_verifier_rejects_wrong_git_revision(tmp_path):
    subprocess.run(("git", "init", str(tmp_path)), check=True)
    subprocess.run(
        ("git", "-C", str(tmp_path), "config", "user.name", "VeriCCL Test"),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(tmp_path), "config", "user.email", "vericcl@example.invalid"),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(tmp_path), "commit", "--allow-empty", "-m", "wrong revision"),
        check=True,
    )

    completed = subprocess.run(
        (sys.executable, str(VERIFY_PATCH), "--source-root", str(tmp_path)),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "pinned MSCCL revision" in completed.stderr


def test_verifier_rejects_dirty_pinned_base_tree(tmp_path, monkeypatch):
    source = tmp_path / "src" / "include" / "msccl.h"
    source.parent.mkdir(parents=True)
    source.write_text("baseline\n", encoding="utf-8")
    subprocess.run(("git", "init", str(tmp_path)), check=True)
    subprocess.run(
        ("git", "-C", str(tmp_path), "config", "user.name", "VeriCCL Test"),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(tmp_path), "config", "user.email", "vericcl@example.invalid"),
        check=True,
    )
    subprocess.run(("git", "-C", str(tmp_path), "add", str(source)), check=True)
    subprocess.run(
        ("git", "-C", str(tmp_path), "commit", "-m", "pinned revision"),
        check=True,
    )
    head = subprocess.run(
        ("git", "-C", str(tmp_path), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source.write_text("modified\n", encoding="utf-8")

    spec = importlib.util.spec_from_file_location("verify_patch", VERIFY_PATCH)
    verifier = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(verifier)
    monkeypatch.setattr(
        verifier,
        "_load_metadata",
        lambda _: {"schema_version": 1, "upstream_commit": head},
    )

    with pytest.raises(ValueError, match="tracked changes"):
        verifier.verify(tmp_path)


def test_verifier_rejects_dirty_patched_tree_outside_hashed_files(
    tmp_path,
    monkeypatch,
):
    for relative in EXPECTED_PATCHED_FILES:
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("patched\n", encoding="utf-8")
    unrelated = tmp_path / "README.md"
    unrelated.write_text("clean\n", encoding="utf-8")
    subprocess.run(("git", "init", str(tmp_path)), check=True)
    subprocess.run(
        ("git", "-C", str(tmp_path), "config", "user.name", "VeriCCL Test"),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(tmp_path), "config", "user.email", "vericcl@example.invalid"),
        check=True,
    )
    subprocess.run(("git", "-C", str(tmp_path), "add", "."), check=True)
    subprocess.run(
        ("git", "-C", str(tmp_path), "commit", "-m", "patched revision"),
        check=True,
    )
    head = subprocess.run(
        ("git", "-C", str(tmp_path), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    unrelated.write_text("dirty\n", encoding="utf-8")

    spec = importlib.util.spec_from_file_location("verify_patch", VERIFY_PATCH)
    verifier = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(verifier)
    patched_files = {
        relative: hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest()
        for relative in EXPECTED_PATCHED_FILES
    }
    monkeypatch.setattr(
        verifier,
        "_load_metadata",
        lambda _: {
            "schema_version": 1,
            "patched_commit": head,
            "patched_files": patched_files,
        },
    )

    with pytest.raises(ValueError, match="tracked changes"):
        verifier.verify(tmp_path, patched_tree=True)


def test_patch_uses_fixed_buffers_and_removes_device_printf_trace():
    patch = PATCH_FILE.read_text(encoding="utf-8")
    added_lines = "\n".join(
        line[1:] for line in patch.splitlines() if line.startswith("+")
    )

    patched_files = tuple(
        re.findall(r"^diff --git a/(\S+) b/\1$", patch, flags=re.MULTILINE)
    )
    assert len(patched_files) == 7
    assert set(patched_files) == {
        "src/collectives/device/msccl_interpreter.h",
        "src/collectives/device/primitives.h",
        "src/collectives/device/prims_ll.h",
        "src/collectives/device/prims_ll128.h",
        "src/collectives/device/prims_simple.h",
        "src/include/msccl.h",
        "src/init.cc",
    }

    assert "#define MSCCL_SLICESTEPS 4" in added_lines
    assert "#define MSCCL_CHUNKSTEPS 4" in added_lines
    assert "VericclRawStepTraceRecord* traceRecords" in added_lines
    assert "unsigned long long* traceRecordCount" in added_lines
    assert "unsigned int* traceOverflow" in added_lines
    assert "atomicAdd" in added_lines
    assert "atomicExch" in added_lines
    assert "traceIndex < traceCapacity" in added_lines
    assert "const bool traceEnabled" in added_lines
    assert "if (traceEnabled)" in added_lines
    assert "src/collectives/device/prims_simple.h" in patch
    assert "src/collectives/device/prims_ll.h" in patch
    assert "src/collectives/device/prims_ll128.h" in patch
    assert '#include "vericcl_trace_format.h"' in added_lines
    assert "__device__ __forceinline__ uint64_t mscclTraceClock()" in added_lines
    assert "__device__ __forceinline__ uint32_t mscclTraceResourceMask(" in added_lines
    assert "&wtime, &traceInfo" not in added_lines
    assert "tb_reach" in added_lines
    assert "dependency_done" in added_lines
    assert "transfer_start" in added_lines
    assert "transfer_end" in added_lines
    assert "rawTrace->iteration = (uint32_t)workIndex;" in added_lines
    assert "rawTrace->iteration = (uint32_t)iter;" not in added_lines
    assert "VERICCL_TRACE_ENABLE" in added_lines
    assert "VERICCL_TRACE_RECORDS" in added_lines
    assert "VERICCL_TRACE_FILE_PREFIX" in added_lines
    assert "VERICCL_EXPECTED_MSCCL_CHUNKSTEPS" in added_lines
    assert "VERICCL_EXPECTED_MSCCL_SLICESTEPS" in added_lines
    assert "vericclCheckStepSignature" in added_lines
    assert "cudaFree(traceInfo->traceRecords)" in added_lines
    assert "cudaFree(traceInfo->traceRecordCount)" in added_lines
    assert "cudaFree(traceInfo->traceOverflow)" in added_lines
    assert 'printf("MSCCLTRACE' not in added_lines
    assert "Rank:%d,Bid:%d,Count:%d" not in added_lines


def test_host_signature_patch_forces_msccl_proxy_steps_only():
    patch = HOST_SIGNATURE_PATCH.read_text(encoding="utf-8")
    added_lines = "\n".join(
        line[1:] for line in patch.splitlines() if line.startswith("+")
    )

    patched_files = tuple(
        re.findall(r"^diff --git a/(\S+) b/\1$", patch, flags=re.MULTILINE)
    )
    assert patched_files == ("src/enqueue.cc",)
    assert "if (info->algorithm == NCCL_ALGO_MSCCL)" in added_lines
    assert "chunkSteps = MSCCL_CHUNKSTEPS;" in added_lines
    assert "sliceSteps = MSCCL_SLICESTEPS;" in added_lines


def test_transfer_start_is_first_captured_inside_every_primitive_protocol():
    patch = PATCH_FILE.read_text(encoding="utf-8")
    added_lines = "\n".join(
        line[1:] for line in patch.splitlines() if line.startswith("+")
    )

    assert re.search(
        r"mscclTraceStart\(\s*VericclRawStepTraceRecord\* trace\)",
        added_lines,
    )
    assert "atomicCAS(" in added_lines
    assert "(unsigned long long*)&trace->transfer_start" in added_lines
    assert "0ULL, timestamp" in added_lines
    assert added_lines.count("mscclTraceStart(trace);") >= 4
    assert "transferStart = mscclTraceClock()" not in added_lines
    assert "rawTrace->transfer_start = transferStart" not in added_lines
    assert "traceRecord->transfer_start = 0;" in added_lines
    assert re.search(
        r"ncclShmem\.mscclShmem\.traceRecord\s*=\s*"
        r"vericclTraceReserve\(traceCommInfo\);",
        added_lines,
    )
    assert "rawTrace = ncclShmem.mscclShmem.traceRecord;" in added_lines
    assert "if (rawTrace->transfer_start == 0)" in added_lines
    assert "atomicExch(traceCommInfo->traceOverflow, 1U);" in added_lines
    for protocol_file in ("prims_simple.h", "prims_ll.h", "prims_ll128.h"):
        protocol_patch = patch.split(
            "diff --git a/src/collectives/device/{}".format(protocol_file), 1
        )[1].split("diff --git a/", 1)[0]
        assert "mscclTraceStart(trace);" in protocol_patch


def test_transfer_start_markers_follow_protocol_readiness_before_data_work():
    patch = PATCH_FILE.read_text(encoding="utf-8")

    simple = patch.split(
        "diff --git a/src/collectives/device/prims_simple.h", 1
    )[1].split("diff --git a/", 1)[0]
    simple_wait = simple.index("waitPeer<")
    simple_start = simple.index("mscclTraceStart(trace);", simple_wait)
    simple_data = simple.index("NPKIT_GPU_PRIMS_WAIT_END(tid);", simple_start)
    assert simple_wait < simple_start < simple_data
    assert "Peer/FIFO/credit readiness is complete" in simple

    ll = patch.split("diff --git a/src/collectives/device/prims_ll.h", 1)[
        1
    ].split("diff --git a/", 1)[0]
    ll_recv = ll.index("peerData = readLL(offset, 0);")
    ll_start = ll.index("mscclTraceStart(trace);", ll_recv)
    ll_reduce = ll.index("data = !SRC ? peerData", ll_start)
    assert ll_recv < ll_start < ll_reduce
    assert re.search(
        r"valid peer FIFO load is the first communication data operation", ll,
        flags=re.IGNORECASE,
    )

    ll128 = patch.split(
        "diff --git a/src/collectives/device/prims_ll128.h", 1
    )[1].split("diff --git a/", 1)[0]
    ll128_ready = ll128.index("NPKIT_GPU_PRIMS_WAIT_END_WITH_SPIN(tid);")
    ll128_start = ll128.index("mscclTraceStart(trace);", ll128_ready)
    ll128_consume = ll128.index("loadRegsFinish(v);", ll128_start)
    assert ll128_ready < ll128_start < ll128_consume
    assert re.search(
        r"valid peer data is now available", ll128, flags=re.IGNORECASE,
    )


def test_trace_initialization_rolls_back_partial_allocations():
    patch = PATCH_FILE.read_text(encoding="utf-8")
    init_patch = patch.split("diff --git a/src/init.cc", 1)[1]
    init_added = "\n".join(
        line[1:] for line in init_patch.splitlines() if line.startswith("+")
    )

    assert "NCCLCHECKGOTO(ncclCudaCalloc(" in init_added
    assert "trace_init_fail:" in init_added
    assert "CUDACHECKIGNORE(cudaFree(traceInfo->traceRecords))" in init_added
    assert "CUDACHECKIGNORE(cudaFree(traceInfo->traceRecordCount))" in init_added
    assert "CUDACHECKIGNORE(cudaFree(traceInfo->traceOverflow))" in init_added
    assert "hostInfo->traceFilePrefix = NULL;" in init_added
    assert "dev_comm_fail:" in init_added
    assert len(re.findall(
        r"NCCLCHECKGOTO\(\s*ncclCudaMemcpy\(", init_added,
    )) >= 2
    assert "vericclTraceDiscard(comm);" in init_added


def test_verifier_copies_every_file_modified_by_the_patches():
    patch_files = (PATCH_FILE, HOST_SIGNATURE_PATCH)
    patched_files = {
        relative
        for patch_file in patch_files
        for relative in re.findall(
            r"^diff --git a/(\S+) b/\1$",
            patch_file.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
    }
    spec = importlib.util.spec_from_file_location("verify_patch", VERIFY_PATCH)
    verifier = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(verifier)

    assert tuple(verifier.PATCH_FILES) == patch_files
    assert patched_files <= set(verifier.REQUIRED_FILES)


def test_patch_reserves_one_record_per_xml_step_before_count_splitting():
    patch = PATCH_FILE.read_text(encoding="utf-8")
    interpreter_patch = patch.split(
        "diff --git a/src/collectives/device/msccl_interpreter.h", 1
    )[1].split("diff --git a/src/collectives/device/primitives.h", 1)[0]

    declaration = "VericclRawStepTraceRecord* rawTrace = NULL;"
    reserve = "vericclTraceReserve(traceCommInfo);"
    dependency_wait = "// first wait if there is a dependence"
    assert interpreter_patch.count(declaration) == 1
    assert interpreter_patch.count(reserve) == 1
    dependency_position = interpreter_patch.index(dependency_wait)
    assert interpreter_patch.index(declaration) < dependency_position
    assert interpreter_patch.index(reserve) < dependency_position


def _write_executable(path, body):
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _run_documented_strategy(command, tmp_path, metadata):
    tool_dir = tmp_path / "bin"
    tool_dir.mkdir()
    command_log = tmp_path / "commands.log"
    _write_executable(
        tool_dir / "git",
        """#!/bin/bash
set -eu
printf 'git %s\n' "$*" >> "$COMMAND_LOG"
if [[ " $* " == *" clone "* ]]; then
  destination="${!#}"
  mkdir -p "$destination/src/include"
elif [[ " $* " == *" rev-parse HEAD "* ]]; then
  printf '%s\n' "$EXPECTED_COMMIT"
fi
""",
    )
    _write_executable(
        tool_dir / "python3",
        """#!/bin/bash
set -eu
printf 'python3 %s\n' "$*" >> "$COMMAND_LOG"
""",
    )
    _write_executable(
        tool_dir / "patch",
        """#!/bin/bash
set -eu
printf 'patch %s\n' "$*" >> "$COMMAND_LOG"
""",
    )
    _write_executable(
        tool_dir / "make",
        """#!/bin/bash
set -eu
printf 'make %s\n' "$*" >> "$COMMAND_LOG"
arguments=" $* "
root=
while (($#)); do
  if [[ "$1" == "-C" ]]; then
    root="$2"
    shift 2
    continue
  fi
  shift
done
if [[ -n "$root" && "$arguments" == *" src.build "* ]]; then
  mkdir -p "$root/build/lib"
fi
""",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "COMMAND_LOG": str(command_log),
            "EXPECTED_COMMIT": metadata["patched_commit"],
            "PATH": "{}:{}".format(tool_dir, environment["PATH"]),
            "TMPDIR": str(tmp_path),
        }
    )
    completed = subprocess.run(
        ("bash", "-eu", "-o", "pipefail", "-c", command),
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )
    return completed, command_log


def test_runtime_readme_installation_strategies_execute_pinned_verifiers(
    tmp_path,
):
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    strategies = dict(
        STRATEGY_PATTERN.findall(RUNTIME_README.read_text(encoding="utf-8"))
    )

    assert set(strategies) == {"strategy-a", "strategy-c"}
    for name, mode in (
        ("strategy-a", "--base-tree"),
        ("strategy-c", "--patched-tree"),
    ):
        case_root = tmp_path / name
        case_root.mkdir()
        completed, command_log = _run_documented_strategy(
            strategies[name],
            case_root,
            metadata,
        )

        assert completed.returncode == 0, completed.stdout + completed.stderr
        log = command_log.read_text(encoding="utf-8")
        assert mode in log
        assert HOST_SIGNATURE_PATCH.name in log
        if name == "strategy-a":
            assert metadata["upstream_repository"] in log
            assert metadata["upstream_commit"] in log
        else:
            assert metadata["fork_repository"] in log
            assert "--branch {}".format(metadata["fork_tag"]) in log
        source_root = case_root / (
            "vericcl-msccl-base"
            if name == "strategy-a"
            else "vericcl-msccl-runtime"
        )
        assert "--source-root {}".format(source_root) in log
        assert (source_root / "build" / "lib").is_dir()


def test_patch_dry_run_and_post_apply_source_scan():
    if not REFERENCE_ROOT or not Path(REFERENCE_ROOT).is_dir():
        pytest.skip("MSCCL reference source is not available")

    reference_root = Path(REFERENCE_ROOT)

    reference_files = (
        reference_root / "src/include/msccl.h",
        reference_root / "src/collectives/device/primitives.h",
        reference_root / "src/collectives/device/prims_ll.h",
        reference_root / "src/collectives/device/prims_ll128.h",
        reference_root / "src/collectives/device/prims_simple.h",
        reference_root / "src/collectives/device/msccl_interpreter.h",
        reference_root / "src/init.cc",
    )
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in reference_files
    }

    completed = subprocess.run(
        (
            sys.executable,
            str(VERIFY_PATCH),
            "--source-root",
            str(reference_root),
        ),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "verification passed" in completed.stdout
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in reference_files
    } == before
