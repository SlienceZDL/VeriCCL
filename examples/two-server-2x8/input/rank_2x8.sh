#!/bin/bash
# One GPU per local rank; optional IB/RoCE selection for each host.
set -euo pipefail
export NCCL_TESTS_DEVICE=${OMPI_COMM_WORLD_LOCAL_RANK}
if [ "${OMPI_COMM_WORLD_RANK}" -lt 8 ]; then
  HCA=${IB_HCA_FIRST:-}
else
  HCA=${IB_HCA_SECOND:-}
fi
if [ -n "$HCA" ]; then export NCCL_IB_HCA="$HCA"; fi
exec "$@"
