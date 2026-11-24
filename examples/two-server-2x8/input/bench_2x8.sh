#!/bin/bash
# Run in-place AllGather with eight ranks per host.
# Usage: bench_2x8.sh <xml[:xml...]> [min_bytes] [max_bytes]
# Required: MPI_HOME, CUDA_HOME, LIB, NCCL_TESTS, HOSTS, NETWORK, SOCKET_IFNAME.
# Optional: IB_HCA_FIRST, IB_HCA_SECOND; CHECK=1 enables correctness checks.
set -euo pipefail
: "${MPI_HOME:?Set MPI_HOME}" "${CUDA_HOME:?Set CUDA_HOME}" "${LIB:?Set LIB}"
: "${NCCL_TESTS:?Set NCCL_TESTS}" "${HOSTS:?Set HOSTS to two comma-separated hosts}"
: "${NETWORK:?Set NETWORK}" "${SOCKET_IFNAME:?Set SOCKET_IFNAME}"
XML=${1:?Pass an XML path or colon-separated list}; MIN=${2:-4194304}; MAX=${3:-2147483648}
IFS=, read -r -a HOST_ARRAY <<< "$HOSTS"
if [ "${#HOST_ARRAY[@]}" -ne 2 ] || [ -z "${HOST_ARRAY[0]}" ] || [ -z "${HOST_ARRAY[1]}" ]; then
  echo 'HOSTS must contain two hosts' >&2; exit 2
fi
CHECK=${CHECK:-0}
if [ "$CHECK" = 1 ]; then ITER=(-w 2 -n 3 -c 1); else ITER=(-w 5 -n 20 -c 0); fi
"$MPI_HOME/bin/mpirun" --bind-to none -np 16 --host "${HOST_ARRAY[0]}:8,${HOST_ARRAY[1]}:8" --map-by ppr:8:node \
  --mca pml ob1 --mca btl self,vader,tcp --mca btl_tcp_if_include "$NETWORK" --mca oob_tcp_if_include "$NETWORK" \
  -x LD_LIBRARY_PATH="$LIB:$MPI_HOME/lib:$CUDA_HOME/lib64" -x NCCL_SOCKET_IFNAME="$SOCKET_IFNAME" \
  -x IB_HCA_FIRST="${IB_HCA_FIRST:-}" -x IB_HCA_SECOND="${IB_HCA_SECOND:-}" \
  -x NCCL_BUFFSIZE=16777216 -x NCCL_PROTO=Simple -x NCCL_ALGO=MSCCL,RING -x MSCCL_XML_FILES="$XML" \
  "$(dirname "$(readlink -f "$0")")/rank_2x8.sh" "$NCCL_TESTS/all_gather_perf" \
  -g 1 -b "$MIN" -e "$MAX" -f 2 -d float "${ITER[@]}" -T 300 < /dev/null
