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

def search_allreduce(args, group):
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
    handle_write_to_directory(Path(), False, lambda: SCCLEncoder().encode(algo), name_sccl_object(name))

    return name, SCCLEncoder().encode(algo)

def search_allgather(args, group):
    # Allgather
    print("search_allgather")

    node_topology = __topologies.create(args)
    topology, route_sketch = parse_and_get_topo_(node_topology, args.sketch_file)
    route_sketch.symmetry.offsets = []
    collective = __collectives.create_sub_coll(args, topology.num_nodes(), "Sub_Allgather", group).chunk_up(route_sketch.hyperparameters.chunkup)
    algo = optimize_allgather_sketch(topology, route_sketch, collective)
    name = str(Collective.AllGather) + '_'.join(map(str, group))

    # output_handler(args, algo, name)
    handle_write_to_directory(Path(), False, lambda: SCCLEncoder().encode(algo), name_sccl_object(name))

    return name, SCCLEncoder().encode(algo)

def search_reduce_scatter(args, group):
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
    handle_write_to_directory(Path(), False, lambda: SCCLEncoder().encode(algo), name_sccl_object(name))

    return name, SCCLEncoder().encode(algo)

def search_broadcast(args, group, chunk_num):
    # Broadcast
    print("search_broadcast")

    node_topology = __topologies.create(args)
    topology, route_sketch = parse_and_get_topo_(node_topology, args.sketch_file)
    route_sketch.symmetry.offsets = []
    collective = __collectives.create_sub_coll(args, topology.num_nodes(),"Sub_Broadcast", group).chunk_up(route_sketch.hyperparameters.chunkup)
    algo = optimize_broadcast_sketch(topology, route_sketch, collective)
    name = str(Collective.Broadcast) + '_'.join(map(str, group))

    # 补充chunk数
    algo_steps = algo.get_steps()
    send_len = len(algo_steps[0].sends)
    for chunk_index in range(1, chunk_num):
        # print(f"chunk_index:{chunk_index}")
        for step in algo_steps:
            for index in range(send_len):
                new_send = step.sends[index][:]
                new_send[0] = chunk_index
                step.sends.append(new_send)

    # output_handler(args, algo, name)
    handle_write_to_directory(Path(), False, lambda: SCCLEncoder().encode(algo), name_sccl_object(name))

    return name, SCCLEncoder().encode(algo)

def task(args, coll, group, chunk_num):
    if coll == Collective.AllReduce:
        return search_allreduce(args, group)
    elif coll == Collective.AllGather:
        return search_allgather(args, group)
    elif coll == Collective.Broadcast:
        return search_broadcast(args, group, chunk_num)
    elif coll == Collective.ReduceScatter:
        return search_reduce_scatter(args, group)
    return None, None

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

        # comm_groups = {
        #     "A": {
        #         "1": [[0, 1, 2, 3, 4, 5, 6, 7],
        #               [8, 9, 10, 11, 12, 13, 14, 15], 
        #               [16, 17, 18, 19, 20, 21, 22, 23],
        #               [24, 25, 26, 27, 28, 29, 30, 31]],
        #         "2": [[0, 8, 16, 24]]
        #     },
        # }

        # opt_, comm_groups, stage1_search_time = search()

        comm_groups = {
            "A": {
                "1": [[0, 1, 2, 3, 4, 5, 6, 7],
                      [8, 9, 10, 11, 12, 13, 14, 15]],
                "2": [[0, 8]]
            },
        }

        # comm_groups = {
        #     "A": {
        #         "0": [[0, 1, 2, 3],
        #               [4, 5, 6, 7]],
        #         "1": [[0, 4]]
        #     },
        # }

        chunk_num = 0
        for comm_group in comm_groups.values():
            for groups in comm_group.values():
                chunk_num = max(chunk_num, len(groups[0]))


        # print(f"comm_groups:{comm_groups}")
        result = ((Collective.AllReduce, 'A', '1'), (Collective.AllReduce, 'A', '2'), (Collective.Broadcast, 'A', '1'))
        # result = ((Collective.AllReduce, 'A', '0'), (Collective.AllReduce, 'A', '1'), (Collective.Broadcast, 'A', '0'))
        

        time_start = time.time()

        algo_jsons = {}
        algo_json = None
        algo_name = None

        stage = []

        # 多进程
        with ProcessPoolExecutor() as executor:
            futures = []
            for reduction in result:
                group_ = comm_groups[reduction[1]][reduction[2]]
                coll_ = reduction[0]
                # if coll_ == Collective.AllReduce:
                    # TODO 当为AllReduce时，只搜索第一个，后面的根据步长修改sends中的[0 1* 2* 3 4 5]即可

                for group in group_:
                    print(f"group:{group}")
                # print(f"group_[0]:{group_[0]}")
                    futures.append(executor.submit(task, args, coll_, group, chunk_num))
                # futures.append(executor.submit(task, args, coll_, group_[0])) # 测试代码

            for future in as_completed(futures):
                algo_name, algo_json = future.result()
                algo_jsons[algo_name] = algo_json

        print(f"algo_jsons_len:{len(algo_jsons)}")

        time_end = time.time()
        stage2_search_time = time_end - time_start
        # print(f"stage1 search time: {stage1_search_time:.6f} s")
        print(f"stage2 search time: {stage2_search_time:.6f} s")
        # print(f"total search time: {stage1_search_time + stage2_search_time:.6f} s")

    return handle
        
    #     # 合成完整json文件
    #     # 加载一个allreduce json模板
    #     allreduce_json = None
    #     with open("Allreduce_template.json", 'r') as f:
    #         allreduce_json = json.load(f)

    #     steps = []
    #     step_num = 0
        
    #     for reduction in result:
    #         group_ = comm_groups[reduction[1]][reduction[2]] # 通信组
    #         coll_ = reduction[0] # 集合通信类型

    #         index = 0
    #         step = None

    #         for group in group_: # 一组 reduction 中的一个
    #             algo_name = str(coll_) + '_'.join(map(str, group))
    #             algo_json = json.loads(algo_jsons[algo_name])
    #             # print(f"{algo_name}: {algo_json} : {type(algo_json)}")

    #             step_ = algo_json["steps"]

    #             # print(step_)

    #             if index == 0:
    #                 step = step_
    #                 stage.append([(step_num + i) for i in range(len(step_))])
    #                 step_num += len(step_)
    #             else:
    #                 for i in range(len(step)):
    #                     print(f"algo_name={algo_name}, len(step)={len(step)}, len(step_={len(step_)}")
    #                     for j in range(len(step_[i]["sends"])):
    #                         step[i]["sends"].append(step_[i]["sends"][j])

    #             index += 1

    #         for s in step:
    #             steps.append(s)

    #         '''
    #         if type(group_) == int: # 完整的一个reduction
    #             print("here")
    #             algo_name = str(coll_) + '_'.join(map(str, group_))
    #             algo_json = json.loads(algo_jsons[algo_name])
    #             step = algo_json["steps"]
    #             for s in step:
    #                 steps.append(s)
    #         else: # 一组 reduction,但是可以同步执行
    #             index = 0
    #             step = None
    #             for group in group_: # 一组 reduction 中的一个
    #                 algo_name = str(coll_) + '_'.join(map(str, group))
    #                 algo_json = json.loads(algo_jsons[algo_name])
    #                 # print(f"{algo_name}: {algo_json} : {type(algo_json)}")
    #                 step_ = algo_json["steps"]
    #                 if index == 0:
    #                     step = step_
    #                 else:
    #                     for i in range(len(step)):
    #                         for j in range(len(step_[i]["sends"])):
    #                             step[i]["sends"].append(step_[i]["sends"][j])

    #                 index += 1

    #             for s in step:
    #                 steps.append(s)
    #         '''

    #     allreduce_json["steps"] = steps
    #     allreduce_json["instance"]["steps"] = len(steps)
    #     allreduce_json["stages"] = stage

    #     print(f"stage: {stage}")

    #     # 保存json文件
    #     with open("Allreduce_4.json", 'w') as f:
    #         json.dump(allreduce_json, f)

    #     time_end = time.time()
    #     stage2_search_time = time_end - time_start
    #     # print(f"stage1 search time: {stage1_search_time:.6f} s")
    #     print(f"stage2 search time: {stage2_search_time:.6f} s")
    #     # print(f"total search time: {stage1_search_time + stage2_search_time:.6f} s")

    # return handle

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