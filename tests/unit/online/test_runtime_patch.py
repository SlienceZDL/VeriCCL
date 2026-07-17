import hashlib
from pathlib import Path
import re
import subprocess
import sys

import pytest


pytestmark = pytest.mark.phase06


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = REPOSITORY_ROOT / "runtime" / "msccl-trace"
FORMAT_HEADER = RUNTIME_ROOT / "include" / "vericcl_trace_format.h"
PATCH_FILE = (
    RUNTIME_ROOT
    / "patches"
    / "0001-vericcl-fixed-step-trace.patch"
)
VERIFY_PATCH = RUNTIME_ROOT / "tools" / "verify_patch.py"
REFERENCE_ROOT = Path("/Users/zdl/work/code/MSCCL_TIME")


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


def test_patch_uses_fixed_buffers_and_removes_device_printf_trace():
    patch = PATCH_FILE.read_text(encoding="utf-8")
    added_lines = "\n".join(
        line[1:] for line in patch.splitlines() if line.startswith("+")
    )

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
    assert "&wtime, &traceInfo" not in added_lines
    assert "tb_reach" in added_lines
    assert "dependency_done" in added_lines
    assert "transfer_start" in added_lines
    assert "transfer_end" in added_lines
    assert "VERICCL_TRACE_ENABLE" in added_lines
    assert "VERICCL_TRACE_RECORDS" in added_lines
    assert "VERICCL_TRACE_FILE_PREFIX" in added_lines
    assert "cudaFree(traceInfo->traceRecords)" in added_lines
    assert "cudaFree(traceInfo->traceRecordCount)" in added_lines
    assert "cudaFree(traceInfo->traceOverflow)" in added_lines
    assert 'printf("MSCCLTRACE' not in added_lines
    assert "Rank:%d,Bid:%d,Count:%d" not in added_lines


def test_patch_dry_run_and_post_apply_source_scan():
    if not REFERENCE_ROOT.is_dir():
        pytest.skip("MSCCL reference source is not available")

    reference_files = (
        REFERENCE_ROOT / "src/include/msccl.h",
        REFERENCE_ROOT / "src/collectives/device/primitives.h",
        REFERENCE_ROOT / "src/collectives/device/prims_simple.h",
        REFERENCE_ROOT / "src/collectives/device/msccl_interpreter.h",
        REFERENCE_ROOT / "src/init.cc",
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
            str(REFERENCE_ROOT),
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
