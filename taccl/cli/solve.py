# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import numpy as np
import os
from taccl.routing import TACCLRouting
from taccl.heuristic_ordering import HeuristicOrderer
from taccl.scheduler import TACCLScheduler
from taccl.reduce_scheduler import TACCLRevScheduler
from .known_collectives import KnownCollectives
from .known_topologies import KnownTopologies
from .common import *
from taccl.p2.p2_search import *
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy

def optimize_allreduce_sketch(topology, route_sketch, collective, allreduce_coll, distribute_over_links=False):
    # allgather
    path_encoder = TACCLRouting(topology, route_sketch, collective)
    orderer = HeuristicOrderer(topology, route_sketch, collective)
    scheduler = TACCLScheduler(topology, route_sketch, collective)

    chunk_send, time_send, chunk_recv, time_recv = path_encoder.optimize_(distribute_over_links)
    time_recv, chunk_recv, switch_time_recv, switch_chunk_recv, switch_time_send, switch_chunk_send, nic_time_recv, nic_chunk_recv, nic_time_send, nic_chunk_send, switch_link_mapping_recv, switch_link_mapping_send, _ = orderer.perform_ordering(
        chunk_send, time_send, chunk_recv, time_recv
    )
    cont_algo_, send_dict = scheduler.optimize_(chunk_recv, time_recv, switch_chunk_recv, switch_time_recv, switch_chunk_send, switch_time_send, nic_chunk_recv, nic_time_recv, nic_chunk_send, nic_time_send, switch_link_mapping_recv, switch_link_mapping_send)
    
    # allreduce
    
    # 补充 allreduce 置为 12
    route_sketch.hyperparameters.heuristic = 12

    orderer = HeuristicOrderer(topology, route_sketch, collective, reverse=True)
    scheduler = TACCLRevScheduler(topology, route_sketch, collective)

    send_dict_base = send_dict

    # heuristic = 12 in routesketch will reverse the chunk order
    time_recv, chunk_order,switch_time_recv, switch_chunk_recv, switch_time_send, switch_chunk_send, nic_time_recv, nic_chunk_recv, nic_time_send, nic_chunk_send, switch_link_mapping_recv, switch_link_mapping_send, paths = orderer.perform_ordering(chunk_send, time_send, chunk_recv, time_recv)
    for r in range(collective.num_nodes):
        for ll in range(len(switch_chunk_recv[r])):
            print("new_swt_recv: ", r, ll, switch_chunk_recv[r][ll])
    for r1 in range(len(chunk_order)):
        for r2 in range(len(chunk_order[r1])):
            for l in range(len(chunk_order[r1][r2])):
                if len(chunk_order[r1][r2][l]):
                    print("old_send_order", r1, r2, chunk_recv[r2][r1][l])
                    print("new_send_order", r2, r1, chunk_order[r1][r2][l])

    ordered_send_dict_reverse = scheduler.optimize_reversed(chunk_order, time_recv, switch_chunk_recv, switch_time_recv, switch_chunk_send, switch_time_send, nic_chunk_recv, nic_time_recv, nic_chunk_send, nic_time_send, switch_link_mapping_recv, switch_link_mapping_send, paths)

    cont_algo = scheduler.build_allreduce_(allreduce_coll,ordered_send_dict_reverse, send_dict_base)
    
    return cont_algo

def optimize_allgather_sketch(topology, route_sketch, collective, distribute_over_links=False):
    path_encoder = TACCLRouting(topology, route_sketch, collective)
    orderer = HeuristicOrderer(topology, route_sketch, collective)
    scheduler = TACCLScheduler(topology, route_sketch, collective)

    chunk_send, time_send, chunk_recv, time_recv = path_encoder.optimize_(distribute_over_links)
    time_recv, chunk_recv, switch_time_recv, switch_chunk_recv, switch_time_send, switch_chunk_send, nic_time_recv, nic_chunk_recv, nic_time_send, nic_chunk_send, switch_link_mapping_recv, switch_link_mapping_send, _ = orderer.perform_ordering(
        chunk_send, time_send, chunk_recv, time_recv
    )
    cont_algo = scheduler.optimize_(chunk_recv, time_recv, switch_chunk_recv, switch_time_recv, switch_chunk_send, switch_time_send, nic_chunk_recv, nic_time_recv, nic_chunk_send, nic_time_send, switch_link_mapping_recv, switch_link_mapping_send)
    return cont_algo[0] # 由于返回值是两个，是元组

def optimize_broadcast_sketch(topology, route_sketch, collective, distribute_over_links=False):
    path_encoder = TACCLRouting(topology, route_sketch, collective)
    orderer = HeuristicOrderer(topology, route_sketch, collective)
    scheduler = TACCLScheduler(topology, route_sketch, collective)

    chunk_send, time_send, chunk_recv, time_recv = path_encoder.optimize_(distribute_over_links)
    time_recv, chunk_recv, switch_time_recv, switch_chunk_recv, switch_time_send, switch_chunk_send, nic_time_recv, nic_chunk_recv, nic_time_send, nic_chunk_send, switch_link_mapping_recv, switch_link_mapping_send, _ = orderer.perform_ordering(
        chunk_send, time_send, chunk_recv, time_recv
    )
    cont_algo = scheduler.optimize_(chunk_recv, time_recv, switch_chunk_recv, switch_time_recv, switch_chunk_send, switch_time_send, nic_chunk_recv, nic_time_recv, nic_chunk_send, nic_time_send, switch_link_mapping_recv, switch_link_mapping_send)
    return cont_algo[0] # 由于返回值是两个，是元组


def optimize_comm_sketch(topology, route_sketch, collective, distribute_over_links=False):
    path_encoder = TACCLRouting(topology, route_sketch, collective)
    orderer = HeuristicOrderer(topology, route_sketch, collective)
    scheduler = TACCLScheduler(topology, route_sketch, collective)

    chunk_send, time_send, chunk_recv, time_recv = path_encoder.optimize(distribute_over_links)
    time_recv, chunk_recv, switch_time_recv, switch_chunk_recv, switch_time_send, switch_chunk_send, nic_time_recv, nic_chunk_recv, nic_time_send, nic_chunk_send, switch_link_mapping_recv, switch_link_mapping_send, _ = orderer.perform_ordering(
        chunk_send, time_send, chunk_recv, time_recv
    )
    cont_algo = scheduler.optimize(chunk_recv, time_recv, switch_chunk_recv, switch_time_recv, switch_chunk_send, switch_time_send, nic_chunk_recv, nic_time_recv, nic_chunk_send, nic_time_send, switch_link_mapping_recv, switch_link_mapping_send)
    return cont_algo


def check_heur_comm_sketch(topology, route_sketch, collective, ts_heur):
    path_encoder = TACCLRouting(topology, route_sketch, collective)
    orderer = HeuristicOrderer(topology, route_sketch, collective)
    scheduler = TACCLScheduler(topology, route_sketch, collective)

    chunk_send, time_send, chunk_recv, time_recv = path_encoder.check_heuristic(ts_heur)
    time_recv, chunk_recv, switch_time_recv, switch_chunk_recv, switch_time_send, switch_chunk_send, nic_time_recv, nic_chunk_recv, nic_time_send, nic_chunk_send, switch_link_mapping_recv, switch_link_mapping_send, _ = orderer.perform_ordering(
        chunk_send, time_send, chunk_recv, time_recv
    )
    cont_algo = scheduler.optimize(chunk_recv, time_recv, switch_chunk_recv, switch_time_recv, switch_chunk_send, switch_time_send, nic_chunk_recv, nic_time_recv, nic_chunk_send, nic_time_send, switch_link_mapping_recv, switch_link_mapping_send)
    return cont_algo

def get_send_dict_base(ts=""):
    assert len(ts)
    return np.load(f"send_dict_{ts}.npy", allow_pickle=True).item()

def process_dict(send_dict_base, topology, collective):
    C = collective.num_chunks
    R = collective.num_nodes
    L = topology.L

    time_recv = [[[[] for l in range(L)] for src in range(R)] for r in range(R)]
    chunk_recv = [[[[] for l in range(L)] for src in range(R)] for r in range(R)]
    time_send = [[[[] for l in range(L)] for src in range(R)] for r in range(R)]
    chunk_send = [[[[] for l in range(L)] for src in range(R)] for r in range(R)]

    for t in send_dict_base:
        for (c,src,r,t_,l) in send_dict_base[t]:
            chunk_send[src][r][l].append(c)
            time_send[src][r][l].append(t_)
            chunk_recv[r][src][l].append(c)
            time_recv[r][src][l].append(t_ + topology.get_invbw(src,r))
    return chunk_send, time_send, chunk_recv, time_recv

def optimize_reduction(reduce_coll, topology, route_sketch, collective, ts, prefer_local_reduce_first=False):
    orderer = HeuristicOrderer(topology, route_sketch, collective, reverse=True)
    scheduler = TACCLRevScheduler(topology, route_sketch, collective)

    send_dict_base = get_send_dict_base(ts)
    chunk_send, time_send, chunk_recv, time_recv = process_dict(send_dict_base, topology, collective)

    # heuristic = 12 in routesketch will reverse the chunk order
    time_recv, chunk_order,switch_time_recv, switch_chunk_recv, switch_time_send, switch_chunk_send, nic_time_recv, nic_chunk_recv, nic_time_send, nic_chunk_send, switch_link_mapping_recv, switch_link_mapping_send, paths = orderer.perform_ordering(chunk_send, time_send, chunk_recv, time_recv)
    for r in range(collective.num_nodes):
        for ll in range(len(switch_chunk_recv[r])):
            print("new_swt_recv: ", r, ll, switch_chunk_recv[r][ll])
    for r1 in range(len(chunk_order)):
        for r2 in range(len(chunk_order[r1])):
            for l in range(len(chunk_order[r1][r2])):
                if len(chunk_order[r1][r2][l]):
                    print("old_send_order", r1, r2, chunk_recv[r2][r1][l])
                    print("new_send_order", r2, r1, chunk_order[r1][r2][l])

    ordered_send_dict_reverse = scheduler.optimize_reversed(chunk_order, time_recv, switch_chunk_recv, switch_time_recv, switch_chunk_send, switch_time_send, nic_chunk_recv, nic_time_recv, nic_chunk_send, nic_time_send, switch_link_mapping_recv, switch_link_mapping_send, paths)
    np.save(f'send_dict_redscat_{ts}.npy', ordered_send_dict_reverse)

    cont_algo = scheduler.build_allreduce(reduce_coll,ordered_send_dict_reverse, send_dict_base, ts)

    return cont_algo

__topologies = None
__collectives = None

def search_allreduce(args, group, reduction_index):
    # Allreduce
    print("search_allreduce")

    node_topology = __topologies.create(args)
    topology, route_sketch = parse_and_get_topo_(node_topology, args.sketch_file)
    route_sketch.symmetry.offsets = []
    # route_sketch.hyperparameters.chunkup = 1 # rank_len / len(group)
    collective = __collectives.create_sub_coll(args, topology.num_nodes(), "Sub_Allgather", group).chunk_up(route_sketch.hyperparameters.chunkup)

    allreduce_coll = __collectives.create_sub_coll(args, topology.num_nodes(), "Sub_Allreduce", group).chunk_up(route_sketch.hyperparameters.chunkup)

    algo = optimize_allreduce_sketch(topology, route_sketch, collective, allreduce_coll)
    name = str(Collective.AllReduce) + '_'.join(map(str, group))

    # output_handler(args, algo, name)
    # handle_write_to_directory(Path(), False, lambda: SCCLEncoder().encode(algo), name_sccl_object(name))

    return name, SCCLEncoder().encode(algo), reduction_index

def search_allgather(args, group, reduction_index):
    # Allgather
    print("search_allgather")

    node_topology = __topologies.create(args)
    topology, route_sketch = parse_and_get_topo_(node_topology, args.sketch_file)
    route_sketch.symmetry.offsets = []
    collective = __collectives.create_sub_coll(args, topology.num_nodes(), "Sub_Allgather", group).chunk_up(route_sketch.hyperparameters.chunkup)
    algo = optimize_allgather_sketch(topology, route_sketch, collective)
    name = str(Collective.AllGather) + '_'.join(map(str, group))

    # output_handler(args, algo, name)
    # handle_write_to_directory(Path(), False, lambda: SCCLEncoder().encode(algo), name_sccl_object(name))

    return name, SCCLEncoder().encode(algo), reduction_index

def search_reduce_scatter(args, group, reduction_index):
    # ReduceScatter
    print("search_reduce_scatter")

    node_topology = __topologies.create(args)
    topology, route_sketch = parse_and_get_topo_(node_topology, args.sketch_file)
    route_sketch.symmetry.offsets = []
    collective = __collectives.create_sub_coll(args, topology.num_nodes(), "Sub_Allgather", group).chunk_up(route_sketch.hyperparameters.chunkup)

    allreduce_coll = __collectives.create_sub_coll(args, topology.num_nodes(), "Sub_Allreduce", group).chunk_up(route_sketch.hyperparameters.chunkup)

    algo = optimize_allreduce_sketch(topology, route_sketch, collective, allreduce_coll)
    name = str(Collective.ReduceScatter) + '_'.join(map(str, group))

    # 移除 Allreduce的 Allgather 部分
    new_algo_steps = []
    algo_steps = algo.get_steps()
    for step in algo_steps:
        if None not in step.sends[0]:
            new_algo_steps.append(step)
    algo.set_steps(new_algo_steps)

    # output_handler(args, algo, name)
    # handle_write_to_directory(Path(), False, lambda: SCCLEncoder().encode(algo), name_sccl_object(name))

    return name, SCCLEncoder().encode(algo), reduction_index

def search_broadcast(args, group, reduction_index):
    # Broadcast
    print("search_broadcast")

    node_topology = __topologies.create(args)
    topology, route_sketch = parse_and_get_topo_(node_topology, args.sketch_file)
    route_sketch.symmetry.offsets = []
    collective = __collectives.create_sub_coll(args, topology.num_nodes(),"Sub_Broadcast", group).chunk_up(route_sketch.hyperparameters.chunkup * topology.num_nodes())
    algo = optimize_broadcast_sketch(topology, route_sketch, collective)
    name = str(Collective.Broadcast) + '_'.join(map(str, group))

    # output_handler(args, algo, name)
    # handle_write_to_directory(Path(), False, lambda: SCCLEncoder().encode(algo), name_sccl_object(name))

    return name, SCCLEncoder().encode(algo), reduction_index

def task(args, coll, group, reduction_index):
    if coll == Collective.AllReduce:
        return search_allreduce(args, group, reduction_index)
    elif coll == Collective.AllGather:
        return search_allgather(args, group, reduction_index)
    elif coll == Collective.Broadcast:
        return search_broadcast(args, group, reduction_index)
    elif coll == Collective.ReduceScatter:
        return search_reduce_scatter(args, group, reduction_index)
    return None, None, reduction_index

def make_handle_search_comm_sketch(cmd_parsers):
    name = 'search'
    cmd = cmd_parsers.add_parser(name)
    global __topologies
    global __collectives
    __topologies = KnownTopologies(cmd)
    __collectives = KnownCollectives(cmd)
    # validate_output_args, output_handler = add_output_sccl_objects(cmd)
    # cmd.add_argument('--sketch-file', type=argparse.FileType('r'))
    cmd.add_argument('--sketch-file', type=str)

    def handle(args, command):
        if command != name:
            return False
        
        # TODO 搜索reduction序列
        # 生成 comm_groups
        # n = 2
        # m = 8
        # comm_groups = generate_comm_groups_single(n, m)

        # opt_, comm_groups, stage1_search_time = search()

        # comm_groups = {
        #     "A": {
        #         "1": [[0, 1, 2, 3, 4, 5, 6, 7],
        #               [8, 9, 10, 11, 12, 13, 14, 15]],
        #         "2": [[0, 8],
        #               [1, 9],
        #               [2, 10],
        #               [3, 11],
        #               [4, 12],
        #               [5, 13],
        #               [6, 14],
        #               [7, 15]]
        #     },
        # }

        comm_groups = {
            "A": {
                "1": [[0, 1, 2, 3, 4, 5, 6, 7],
                      [8, 9, 10, 11, 12, 13, 14, 15],
                      [16, 17, 18, 19, 20, 21, 22, 23],
                      [24, 25, 26, 27, 28, 29, 30, 31]],
                "2": [[0, 8],
                      [1, 9],
                      [2, 10],
                      [3, 11],
                      [4, 12],
                      [5, 13],
                      [6, 14],
                      [7, 15]]
            },
        }

        chunk_num = 0
        for comm_group in comm_groups.values():
            for groups in comm_group.values():
                chunk_num = max(chunk_num, len(groups[0]))

        # result = ((Collective.AllReduce, 'A', '1'), (Collective.AllReduce, 'A', '2'), (Collective.Broadcast, 'A', '1'))
        
        # 4 阶段
        # 2*2
        # work_list = [
        #              [ Collective.ReduceScatter, [[0, 1], [2, 3]]],
        #              [ Collective.ReduceScatter, [[0, 2], [1, 3]]],
        #              [ Collective.AllGather, [[0, 2], [1, 3]]],
        #              [ Collective.AllGather, [[0, 1], [2, 3]]]
        #             ]

        # 2*4
        # work_list = [
        #              [ Collective.ReduceScatter, [[0, 1, 2, 3], [4, 5, 6, 7]]],
        #              [ Collective.ReduceScatter, [[0, 4], [1, 5], [2, 6], [3, 7]]],
        #              [ Collective.AllGather, [[0, 4], [1, 5], [2, 6], [3, 7]]],
        #              [ Collective.AllGather, [[0, 1, 2, 3], [4, 5, 6, 7]]]
        #             ]
        
        # 4*4
        # work_list = [
        #              [ Collective.ReduceScatter, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]],
        #              [ Collective.ReduceScatter, [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]],
        #              [ Collective.AllGather, [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]],
        #              [ Collective.AllGather, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]]
        #             ]

        # 2*8
        # work_list = [
        #              [ Collective.ReduceScatter, [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]],
        #              [ Collective.ReduceScatter, [[0, 8], [1, 9], [2, 10], [3, 11], [4, 12], [5, 13], [6, 14], [7, 15]]],
        #              [ Collective.AllGather, [[0, 8], [1, 9], [2, 10], [3, 11], [4, 12], [5, 13], [6, 14], [7, 15]]],
        #              [ Collective.AllGather, [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]]
        #             ]

        # 4*8
        work_list = [
                     [ Collective.ReduceScatter, [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]],
                     [ Collective.ReduceScatter, [[0, 8, 16, 24], [1, 9, 17, 25], [2, 10, 18, 26], [3, 11, 19, 27], [4, 12, 20, 28], [5, 13, 21, 29], [6, 14, 22, 30], [7, 15, 23, 31]]],
                     [ Collective.AllGather, [[0, 8, 16, 24], [1, 9, 17, 25], [2, 10, 18, 26], [3, 11, 19, 27], [4, 12, 20, 28], [5, 13, 21, 29], [6, 14, 22, 30], [7, 15, 23, 31]]],
                     [ Collective.AllGather, [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]]
                    ]

        # 8*8
        # work_list = [
        #                 [
        #                     Collective.ReduceScatter, 
        #                     [
        #                         [0, 1, 2, 3, 4, 5, 6, 7], 
        #                         [8, 9, 10, 11, 12, 13, 14, 15], 
        #                         [16, 17, 18, 19, 20, 21, 22, 23], 
        #                         [24, 25, 26, 27, 28, 29, 30, 31],
        #                         [32, 33, 34, 35, 36, 37, 38, 39], 
        #                         [40, 41, 42, 43, 44, 45, 46, 47], 
        #                         [48, 49, 50, 51, 52, 53, 54, 55], 
        #                         [56, 57, 58, 59, 60, 61, 62, 63]
        #                     ]
        #                 ],
        #                 [
        #                     Collective.ReduceScatter, 
        #                     [
        #                         [0, 8, 16, 24, 32, 40, 48, 56], 
        #                         [1, 9, 17, 25, 33, 41, 49, 57], 
        #                         [2, 10, 18, 26, 34, 42, 50, 58], 
        #                         [3, 11, 19, 27, 35, 43, 51, 59], 
        #                         [4, 12, 20, 28, 36, 44, 52, 60], 
        #                         [5, 13, 21, 29, 37, 45, 53, 61], 
        #                         [6, 14, 22, 30, 38, 46, 54, 62], 
        #                         [7, 15, 23, 31, 39, 47, 55, 63]
        #                     ]
        #                 ],
        #                 [
        #                     Collective.AllGather, 
        #                     [
        #                         [0, 8, 16, 24, 32, 40, 48, 56], 
        #                         [1, 9, 17, 25, 33, 41, 49, 57], 
        #                         [2, 10, 18, 26, 34, 42, 50, 58], 
        #                         [3, 11, 19, 27, 35, 43, 51, 59], 
        #                         [4, 12, 20, 28, 36, 44, 52, 60], 
        #                         [5, 13, 21, 29, 37, 45, 53, 61], 
        #                         [6, 14, 22, 30, 38, 46, 54, 62], 
        #                         [7, 15, 23, 31, 39, 47, 55, 63]
        #                     ]
        #                 ],
        #                 [
        #                     Collective.AllGather, 
        #                     [
        #                         [0, 1, 2, 3, 4, 5, 6, 7], 
        #                         [8, 9, 10, 11, 12, 13, 14, 15], 
        #                         [16, 17, 18, 19, 20, 21, 22, 23], 
        #                         [24, 25, 26, 27, 28, 29, 30, 31],
        #                         [32, 33, 34, 35, 36, 37, 38, 39], 
        #                         [40, 41, 42, 43, 44, 45, 46, 47], 
        #                         [48, 49, 50, 51, 52, 53, 54, 55], 
        #                         [56, 57, 58, 59, 60, 61, 62, 63]
        #                     ]
        #                 ]
        # ]

        # 3 阶段

        # 2*4
        # work_list = [
        #              [ Collective.AllReduce, [[0, 1, 2, 3], [4, 5, 6, 7]]],
        #              [ Collective.AllReduce, [[0, 4]]],
        #              [ Collective.Broadcast, [[0, 1, 2, 3], [4, 5, 6, 7]]]
        #             ]
        
        # 4*4
        # work_list = [
        #              [ Collective.AllReduce, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]],
        #              [ Collective.AllReduce, [[0, 4, 8, 12]]],
        #              [ Collective.Broadcast, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]]
        #             ]

        # 2*8
        # work_list = [
        #              [ Collective.AllReduce, [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]],
        #              [ Collective.AllReduce, [[0, 8]]],
        #              [ Collective.Broadcast, [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]]
        #             ]

        # 4*8
        # work_list = [
        #              [ Collective.AllReduce, [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]],
        #              [ Collective.AllReduce, [[0, 8, 16, 24]]],
        #              [ Collective.Broadcast, [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]]
        #             ]

        # 2 阶段

        # 2*4
        # work_list = [
        #              [ Collective.AllReduce, [[0, 1, 2, 3], [4, 5, 6, 7]]],
        #              [ Collective.AllReduce, [[0, 4], [1, 5], [2, 6], [3, 7]]]
        #             ]
        
        # 4*4
        # work_list = [
        #              [ Collective.AllReduce, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]],
        #              [ Collective.AllReduce, [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]]
        #             ]

        # 2*8
        # work_list = [
        #              [ Collective.AllReduce, [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]],
        #              [ Collective.AllReduce, [[0, 8], [1, 9], [2, 10], [3, 11], [4, 12], [5, 13], [6, 14], [7, 15]]]
        #             ]

        # 4*8
        # work_list = [
        #              [ Collective.AllReduce, [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]],
        #              [ Collective.AllReduce, [[0, 8, 16, 24], [1, 9, 17, 25], [2, 10, 18, 26], [3, 11, 19, 27], [4, 12, 20, 28], [5, 13, 21, 29], [6, 14, 22, 30], [7, 15, 23, 31]]]
        #             ]

        # 2 阶段
        # 2*4
        # work_list = [
        #              [ Collective.ReduceScatter, [[0, 1, 2, 3], [4, 5, 6, 7]]],
        #              [ Collective.AllGather, [[0, 4], [1, 5], [2, 6], [3, 7]]]
        #             ]
        
        # 4*4
        # work_list = [
        #              [ Collective.AllReduce, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]],
        #              [ Collective.AllReduce, [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]]
        #             ]

        # 2*8
        # work_list = [
        #              [ Collective.AllReduce, [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]]],
        #              [ Collective.AllReduce, [[0, 8], [1, 9], [2, 10], [3, 11], [4, 12], [5, 13], [6, 14], [7, 15]]]
        #             ]

        # 4*8
        # work_list = [
        #              [ Collective.AllReduce, [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]],
        #              [ Collective.AllReduce, [[0, 8, 16, 24], [1, 9, 17, 25], [2, 10, 18, 26], [3, 11, 19, 27], [4, 12, 20, 28], [5, 13, 21, 29], [6, 14, 22, 30], [7, 15, 23, 31]]]
        #             ]
        
        time_start = time.time()

        algo_jsons = {}
        algo_json = None
        algo_name = None

        stage = []


        def rs_chunk_stage1(algo_json):
            # 获取到steps
            steps = algo_json["steps"]

            new_steps = []

            for i in range(len(steps)):
                new_step1 = copy.deepcopy(steps[i]) # 操作之前复制
                new_step2 = copy.deepcopy(steps[i]) # 操作之前复制
                new_step3 = copy.deepcopy(steps[i]) # 操作之前复制

                # 对原先的step的每一个send的第一个chunk_id进行×2
                sends =  steps[i]["sends"]
                for j in range(len(sends)):
                    sends[j][0] = sends[j][0] * 4 # 或者是4

                # 对拷贝的step的每一个send的第一个chunk_id进行×2 + 1
                new_sends1 =  new_step1["sends"]
                for j in range(len(new_sends1)):
                    new_sends1[j][0] = new_sends1[j][0] * 4 + 1

                new_sends2 =  new_step2["sends"]
                for j in range(len(new_sends2)):
                    new_sends2[j][0] = new_sends2[j][0] * 4 + 2

                new_sends3 =  new_step3["sends"]
                for j in range(len(new_sends3)):
                    new_sends3[j][0] = new_sends3[j][0] * 4 + 3

                new_steps.append(steps[i])
                new_steps.append(new_step1)
                new_steps.append(new_step2)
                new_steps.append(new_step3)

            algo_json["steps"] = new_steps

            return json.dumps(algo_json)
        
        def ar_chunk(algo_json, group_size):
            # 获取到steps
            steps = algo_json["steps"]

            new_steps = []

            epoch_num = algo_json["instance"]["chunks"] // group_size
            
            
            for i in range(len(steps)):
                new_steps.append(steps[i]) # 对原先的step不变
                for epoch in range(epoch_num-1): # 复制的份数
                    new_step = copy.deepcopy(steps[i]) # 操作之前复制
                    
                    # 对拷贝的step的每一个send的第一个chunk_id进行重新映射     ranks // len(group) = 2
                    new_sends =  new_step["sends"]
                    for j in range(len(new_sends)):
                        new_sends[j][0] = new_sends[j][0] + (epoch+1) * group_size
                    new_steps.append(new_step)

            algo_json["steps"] = new_steps

            return json.dumps(algo_json)
        
        # 多进程
        with ProcessPoolExecutor() as executor:
            futures = []
            for index, reduction in enumerate(work_list):
                group_list = reduction[1]
                coll = reduction[0]
                group = group_list[0] # 只搜索第一个, 其余根据规则映射

                print(f"group:{group}, coll:{coll}, index:{index}")
                futures.append(executor.submit(task, args, coll, group, index))


            for future in as_completed(futures):
                algo_name, algo_json, index = future.result()
                print(f"index:{index}")

                reduction = work_list[index]
                group_list = reduction[1]

                if reduction[0] == Collective.ReduceScatter or reduction[0] == Collective.AllGather:
                    # TODO 完成ReduceScatter的chunk重新分配
                    if index == 0 or index == 3:
                        algo_json = rs_chunk_stage1(json.loads(algo_json))
                    elif index == 1 or index == 2 :
                        # algo_json = rs_chunk_stage2(json.loads(algo_json))
                        pass
                if reduction[0] == Collective.AllReduce:
                    # TODO 完成AllReduce的chunk重新分配
                    algo_json = ar_chunk(json.loads(algo_json), len(group_list[0]))
                
                algo_jsons[algo_name] = algo_json

                for i, group in enumerate(group_list): # 映射剩余的子算子
                    if i == 0: # 跳过第一个
                        n = str(reduction[0]) + '_'.join(map(str, group))
                        print(f"algo_name:{algo_name}, tmp:{n}, index:{index}")
                        assert algo_name == n

                        # TODO 测试
                        with (Path() / (algo_name+".json")).open('w') as f:
                            json.dump(json.loads(algo_json), f)

                        continue
                    new_algo = copy.deepcopy(algo_json) # 复制一份文件
                    new_algo_json = json.loads(new_algo)

                    if reduction[0] == Collective.ReduceScatter or reduction[0] == Collective.AllGather:
                        if index == 0 or index == 3:
                            # 计算 rank id 步长
                            step_len = group_list[1][0] - group_list[0][0]
                            steps = new_algo_json["steps"]
                            for step in steps:
                                sends = step["sends"]
                                for send in sends: # 更新rank id
                                    send[1] = send[1] + step_len * i
                                    send[2] = send[2] + step_len * i

                            new_algo_name = str(reduction[0]) + '_'.join(map(str, group))
                            algo_jsons[new_algo_name] = json.dumps(new_algo_json)
                        elif index == 1 or index == 2:
                            # 计算 rank id 步长
                            step_len = group_list[1][0] - group_list[0][0]
                            steps = new_algo_json["steps"]
                            for step in steps:
                                sends = step["sends"]
                                for send in sends: # 更新 rank id , chunk id -> i 从 1 开始, +的2是 16 / 8
                                    send[0] = send[0] + i * 2 # 2表示reducescatter的分组数量
                                    send[1] = send[1] + step_len * i
                                    send[2] = send[2] + step_len * i
                            new_algo_name = str(reduction[0]) + '_'.join(map(str, group))
                            algo_jsons[new_algo_name] = json.dumps(new_algo_json)
                    
                    if reduction[0] == Collective.AllReduce or reduction[0] == Collective.Broadcast:
                        # 计算 rank id 步长
                        step_len = group_list[1][0] - group_list[0][0]
                        steps = new_algo_json["steps"]
                        for step in steps:
                            sends = step["sends"]
                            for send in sends: # 更新 rank id
                                send[1] = send[1] + step_len * i
                                send[2] = send[2] + step_len * i
                        new_algo_name = str(reduction[0]) + '_'.join(map(str, group))
                        algo_jsons[new_algo_name] = json.dumps(new_algo_json)

                    # TODO 测试
                    with (Path() / (new_algo_name+".json")).open('w') as f:
                        json.dump(new_algo_json, f)

        print(f"algo_jsons_len:{len(algo_jsons)}")

    # return handle
        
        # 合成完整json文件
        # 加载一个allreduce json模板
        allreduce_json = None
        with open("Allreduce_template.json", 'r') as f:
            allreduce_json = json.load(f)

        steps = []
        step_num = 0
        
        for reduction in work_list:
            group_ = reduction[1] # 通信组
            coll_ = reduction[0] # 集合通信类型

            index = 0
            step = None

            for group in group_: # 一组 reduction 中的一个
                algo_name = str(coll_) + '_'.join(map(str, group))
                algo_json = json.loads(algo_jsons[algo_name])
                # print(f"{algo_name}: {algo_json} : {type(algo_json)}")

                step_ = algo_json["steps"]

                # print(step_)

                if index == 0:
                    step = step_
                    stage.append([(step_num + i) for i in range(len(step_))])
                    step_num += len(step_)
                else:
                    for i in range(len(step)):
                        # print(f"algo_name={algo_name}, len(step)={len(step)}, len(step_={len(step_)}")
                        for j in range(len(step_[i]["sends"])):
                            step[i]["sends"].append(step_[i]["sends"][j])

                index += 1

            for s in step:
                steps.append(s)

        allreduce_json["steps"] = steps
        allreduce_json["instance"]["steps"] = len(steps)
        allreduce_json["stages"] = stage

        print(f"stage: {stage}")

        # 保存json文件
        with open("Allreduce_16.json", 'w') as f:
            json.dump(allreduce_json, f)

        time_end = time.time()
        stage2_search_time = time_end - time_start
        # print(f"stage1 search time: {stage1_search_time:.6f} s")
        print(f"stage2 search time: {stage2_search_time:.6f} s")
        # print(f"total search time: {stage1_search_time + stage2_search_time:.6f} s")

    return handle

def  make_handle_solve_comm_sketch(cmd_parsers):
    name = 'solve'
    cmd = cmd_parsers.add_parser(name)
    topologies = KnownTopologies(cmd)
    collectives = KnownCollectives(cmd)
    validate_output_args, output_handler = add_output_sccl_objects(cmd)
    # cmd.add_argument('--topo-file', type=argparse.FileType('r'))
    cmd.add_argument('--sketch-file', type=argparse.FileType('r'))
    # cmd.add_argument('--topo-name', type=str)
    cmd.add_argument('--ts-heur', type=int, default="-1")
    def handle(args, command):
        if command != name:
            return False

        time_start = time.time()

        validate_output_args(args)
        node_topology = topologies.create(args) # 解析topo文件,获得单节点内部拓扑结构
        topology, route_sketch = parse_and_get_topo(node_topology, args.sketch_file) # 解析sketch文件,构造整个拓扑
        collective = collectives.create(args, topology.num_nodes()).chunk_up(route_sketch.hyperparameters.chunkup)
        ts_heur = args.ts_heur
        if ts_heur == -1:
            algo = optimize_comm_sketch(topology, route_sketch, collective)
        else:
            algo = check_heur_comm_sketch(topology, route_sketch, collective, ts_heur)
        output_handler(args, algo, algo.name + "_taccl")

        time_end = time.time()
        print(f"solve time: {time_end - time_start:.6f} s")
        
        return True
    
    return handle

def make_handle_combine_comm_sketch(cmd_parsers):
    name = 'combine'
    cmd = cmd_parsers.add_parser(name)
    topologies = KnownTopologies(cmd)
    collectives = KnownCollectives(cmd)
    validate_output_args, output_handler = add_output_sccl_objects(cmd)
    cmd.add_argument('--sketch-file', type=str, default=None)
    cmd.add_argument('--ts', type=str, help='timestamp of send_dict for Allgather')
    cmd.add_argument('--prefer-local-reduce-first', action='store_true', help='should prefer reducing a chunk locally first if it is the same either way')
    def handle(args, command):
        if command != name:
            return False
        if args.sketch_file is None:
            cmd_parsers.error('Must specify sketch file')

        time_start = time.time()

        assert os.path.isfile(args.sketch_file), "sketch file does not exist"
        sketch_file = open(args.sketch_file, 'r')

        validate_output_args(args)
        node_topology = topologies.create(args)
        topology, route_sketch = parse_and_get_topo(node_topology, sketch_file, reduce=True)
        collective = collectives.create(args, topology.num_nodes()).chunk_up(route_sketch.hyperparameters.chunkup)

        import copy
        new_args = copy.deepcopy(args)
        # new_args.collective = 'Allgather'
        # new_args.collective = 'ReduceScatter'
        # new_args.collective = 'Reduce'
        new_args.collective = 'Allreduce'
        # new_args.collective = 'Sub_Broadcast'
        allreduce_coll = collectives.create(new_args, topology.num_nodes()).chunk_up(route_sketch.hyperparameters.chunkup)
        algo = optimize_reduction(allreduce_coll, topology, route_sketch, collective, args.ts, args.prefer_local_reduce_first)
        output_handler(args, algo, algo.name + "_taccl")
        
        time_end = time.time()
        print(f"combine time: {time_end - time_start:.6f} s")

        return True
    

    return handle