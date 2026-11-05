#!/usr/bin/env bash
set -euo pipefail
task_root=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
task_python=${VERICCL_PYTHON:-"$task_root/.venv/bin/python"}
exec "$task_python" "$task_root/runtime/build_runtime.py" "$@"
