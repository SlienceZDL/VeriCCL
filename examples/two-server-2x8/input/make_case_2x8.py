"""Write inputs for in-place AllGather on two servers with eight GPUs each.

Adapt make_topology_2x8.py to the target hardware. Supply runtime, MPI,
host, and network settings explicitly; paths must match on both hosts.
"""
import argparse
import json
from pathlib import Path

from make_topology_2x8 import build


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', required=True, type=Path, help='directory for the four JSON files')
    parser.add_argument('--runtime', required=True, type=Path,
                        help='directory with bin/vericcl-probe, bin/vericcl-native and lib/libnccl.so (built MSCCL)')
    parser.add_argument('--mpi-home', required=True)
    parser.add_argument('--cuda-home', default='/usr/local/cuda-12.8')
    parser.add_argument('--hosts', required=True, help='two hosts; ranks 0-7 on the first')
    parser.add_argument('--network', required=True, help='subnet MPI uses between the hosts')
    parser.add_argument('--socket-ifname', required=True)
    parser.add_argument('--ib-hca', default='', help='optional comma-separated IB/RoCE device names')
    parser.add_argument('--port', type=int, default=39517, help='control port on the first host')
    parser.add_argument('--total-bytes', type=int, default=16 << 20, help='input bytes per rank (M)')
    parser.add_argument('--slice-bytes', type=int, default=4 << 20, help='slice size S; M must be a multiple')
    args = parser.parse_args()
    args.runtime = args.runtime.expanduser().resolve()
    args.mpi_home = str(Path(args.mpi_home).expanduser().resolve())
    args.cuda_home = str(Path(args.cuda_home).expanduser().resolve())
    hosts = [host.strip() for host in args.hosts.split(',')]
    if len(hosts) != 2 or not all(hosts) or hosts[0] == hosts[1]:
        raise SystemExit('--hosts needs exactly two hosts')
    lib = '%s/lib:%s/lib:%s/lib64' % (args.runtime, args.mpi_home, args.cuda_home)
    documents = {
        'topology': build(),
        'atom': {'synthesis_method': 'hierarchy', 'strategies': {}},
        'sketch': {
            'collective': {'operator': 'allgather', 'datatype': 'float32', 'inplace': True},
            'hyperparameters': {'total_size_bytes': args.total_bytes, 'slice_size_bytes': args.slice_bytes},
            'solver': {'backend': 'scip', 'cpu_budget': 8, 'max_channels': 8, 'channel_candidates': [4],
                       'max_parallel_models': 1, 'max_threads_per_model': 8, 'per_model_timeout_s': 120,
                       'solver_seed': 0},
            'budgets': {'cleanup_reserve_s': 2, 'max_initial_synthesis_s': 300, 'max_run_wall_s': 10800,
                        'max_tuning_rounds': 6, 'max_repair_per_round_s': 180, 'max_repair_total_s': 900,
                        'max_offline_validation_per_round_s': 180, 'max_offline_validation_total_s': 720,
                        'max_online_validation_per_round_s': 3600, 'max_online_validation_total_s': 7200},
            'tolerances': {'absolute': 1e-5, 'relative': 1e-5},
            'calibration': {'enabled': True, 'equivalence_classes': True, 'max_relative_fit_error': 0.1,
                            'job_timeout_s': 1200, 'launches': 10},
            'trace_and_measure': {'enabled': True, 'job_timeout_s': 1200, 'trace_capacity_bytes': 3538984},
            'runtime_compatibility': {'enabled': True},
            'history': {'calibration_reuse': True, 'calibration_index': 'calibration-cache/index.json',
                        'calibration_max_age_s': 604800, 'schedule_reuse': False,
                        'schedule_index': 'schedule-history/index.json',
                        'relative_bandwidth_tolerance': 0.5, 'max_schedule_candidates': 4},
        },
        'runtime': {
            'protocol': 'Simple', 'tb_layout': 'separate',
            'device_memory_budget_bytes': 15 << 30, 'scratch_bytes_per_rank': 512 << 20,
            'rank_map': [{'rank': rank, 'host': hosts[rank // 8], 'device': rank % 8} for rank in range(16)],
            'paths': {'cuda_home': args.cuda_home, 'mpi_bin': args.mpi_home + '/bin',
                      'vericcl_driver': str(args.runtime / 'bin/vericcl-probe'),
                      'vericcl_native': str(args.runtime / 'bin/vericcl-native')},
            'environment': {'LD_LIBRARY_PATH': lib, 'NCCL_BUFFSIZE': '16777216', 'NCCL_PROTO': 'Simple',
                            'NCCL_IB_HCA': args.ib_hca, 'NCCL_SOCKET_IFNAME': args.socket_ifname,
                            'OMPI_MCA_pml': 'ob1', 'OMPI_MCA_btl': 'self,vader,tcp',
                            'OMPI_MCA_btl_tcp_if_include': args.network,
                            'OMPI_MCA_oob_tcp_if_include': args.network},
            'control': {'bind_address': hosts[0], 'port': args.port},
        },
    }
    if not args.ib_hca:
        documents['runtime']['environment'].pop('NCCL_IB_HCA')
    args.output.mkdir(parents=True, exist_ok=True)
    for name, document in documents.items():
        (args.output / (name + '.json')).write_text(json.dumps(document, indent=1) + '\n')
    print('wrote', ', '.join(str(args.output / (name + '.json')) for name in documents))


if __name__ == '__main__':
    main()
