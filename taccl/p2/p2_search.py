# from taccl.semantic.common_enum import Collective
# from taccl.semantic.reduction import Reduction
from taccl.p2.p2_enum import Collective
from taccl.p2.p2_reduction import Reduction
from taccl.dsl.dsl_yacc import *
import itertools
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def generate_sequences(max_len, comm_groups):
    elements = []
    collectives = [Collective.AllReduce, Collective.AllGather, Collective.Broadcast, Collective.ReduceScatter]
    # collectives = [Collective.ReduceScatter, Collective.AllGather]
    for collective in collectives:
        for group_name, group_types in comm_groups.items():
            for comm_type in group_types.keys():
                elements.append((collective, group_name, comm_type))
    sequences = []
    for r in range(1, max_len + 1):
        sequences.extend(itertools.product(elements, repeat=r))
    print("seq_len : "+ str(len(sequences))) # （coll_num * comm_num）^ seq_len
    return sequences

def check_sequence(reduction, s, comm_groups):
    for op, group_name, comm_type in s:
        groups = comm_groups[group_name][comm_type]
        for group in groups:
            if not reduction.check_and_update(op, group):
                return False
    return reduction.check_allreduce()

def search_task(rank_num, seq_chunk, comm_groups):
    local_result = []
    reduction = Reduction(rank_num)
    for s in seq_chunk:
        reduction.init_matrix()
        if check_sequence(reduction, s, comm_groups):
            local_result.append(s)
    
    return local_result

def search():
    start_time = time.time()

    # 解析DSL
    lexer = lex.lex()
    parser = yacc.yacc()

    with open('/home/zy/Canvas/taccl/dsl/dsl/test.dsl', 'r') as file:
        while True:
            intent_line = file.readline()
            if intent_line:
                r = parser.parse(intent_line)
            else:
                break

    define = {}
    link_type = {}
    intra_node = {}
    inter_node = {}
    for index, def_type in enumerate(define_v.lvalues):
        define[def_type] = define_v.rvalues[index]
    for index, linktype in enumerate(linktype_v.lvalues):
        link_type[linktype] = linktype_v.rvalues[index]
    for index, intra in enumerate(intra_v.lvalues):
        intra_node[intra] = intra_v.rvalues[index]
    for index, inter in enumerate(inter_v.lvalues):
        inter_node[inter] = inter_v.rvalues[index]
    # END

    comm_groups = generate_comm_groups(define["rtsw"], define["nnode"] // define["rtsw"], define["ngpu_per_node"])

    comm_groups = {
            "A": {
                "0": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]],
                "1": [[0, 1, 2, 3, 4, 5, 6, 7],
                      [8, 9, 10, 11, 12, 13, 14, 15]],
                "2": [[0, 8]],
                "3": [[0, 8],
                      [1, 9],
                      [2, 10],
                      [3, 11],
                      [4, 12],
                      [5, 13],
                      [6, 14],
                      [7, 15]]
            },
        }
    # GPU数量
    gpu_num = define["ngpu_per_node"] * define["nnode"] # 8 * 2 = 16

    # 构建带宽延迟矩阵 [16*16]
    matrix_bandwidth_delay = [[tuple((0, 0)) for _ in range(gpu_num)] for _ in range(gpu_num)]

    # 先构建节点内的
    # {switch => [(0,1,2,3,4,5,6,7)->(nvlink,1)]}
    # intra_v.lvalues ['intra_node_bw_delay']
    # intra_v.rvalues [[{'conn_type': 'switch', 'tuple': [0, 1, 2, 3], 'linktype': 'nvlink', 'number': 2}]]
    intra_node_bw_delay = intra_node["intra_node_bw_delay"][0]
    if "switch" == intra_node_bw_delay['conn_type']:
        group = intra_node_bw_delay['tuple']
        linktype = intra_node_bw_delay['linktype']
        number = intra_node_bw_delay['number']
        for n in range(define['nnode']):
            for i in group:
                for j in group:
                    if i == j:
                        continue
                    matrix_bandwidth_delay[n * define['ngpu_per_node'] + i][n * define['ngpu_per_node'] + j] = (link_type[linktype][0] * number, link_type[linktype][1])
    

    # print_matrix(matrix_bandwidth_delay)
    # 再构建节点间的
    # {'conn_type': 'match', 'tuple': [0, 1, 2, 3]}
    inter_node_bw_delay = inter_node["inter_node_bw_delay"]
    if "match" == inter_node_bw_delay['conn_type']:
        group = inter_node_bw_delay['tuple']
        if define['nnic_per_node'] == define['ngpu_per_node']: # 每个GPU一个NIC
            for n in range(define['nnode']):
                for x in range(define['nnode'] - 1):
                    for i in group:
                        for j in group:
                            matrix_bandwidth_delay[((n + x + 1) % define['nnode']) * define['ngpu_per_node'] + i][n * define['ngpu_per_node'] + j] = (link_type['intra_rtsw'][0], link_type['intra_rtsw'][1])
        elif define['nnic_per_node'] == 1: # 所有GPU一个NIC
            pass
            
    # print("-------------------------------------------------")
    # print_matrix(matrix_bandwidth_delay)
    # END

    seq_len = 4
    
    # 实现 cost model 搜索算法 
    sequences = generate_sequences(seq_len, comm_groups)
    
    result = []

    opt_comm_op = None
    min_cost = float('inf')
    
    epoch_num = 4

    gap = 1000
    num_chunks = len(sequences) // gap
    chunks = [sequences[i*gap : (i+1)*gap] for i in range(num_chunks + 1)]
    
    with ProcessPoolExecutor(max_workers=64) as executor:
        futures = [executor.submit(search_task, gpu_num, chunk, comm_groups) for chunk in chunks]
        for future in as_completed(futures):
            result.extend(future.result())

    # result = search_task(gpu_num, sequences, comm_groups)
    
    end_time = time.time()
    
    # print("------------------result------------------")
    for r in result:
        print(r)
        # TODO 计算cost
        cur_cost = comm_list_cost(matrix_bandwidth_delay, r, comm_groups, epoch_num)
        if cur_cost < min_cost:
            opt_comm_op = r
            min_cost = cur_cost
        print(f"cur_cost: {cur_cost}, min_cost: {min_cost}")
    # print("-------------------end--------------------")
    
    search_time = end_time - start_time
    print(f"stage1 search time: {search_time:.6f} s")
    
    # TODO: 返回最优结果
    print("opt_comm_op:", opt_comm_op)
    print(f"sequences:{len(sequences)}")
    print(f"result len:{len(result)}")
    result_dict = {}
    for i in range(seq_len):
        result_dict[i+1] = []

    for r in result:
        for i in range(seq_len):
            if (i+1) == len(r):
                result_dict[i+1].append(r)

    print("长度为2的序列数量: ", len(result_dict[2]))
    # print_result(result, comm_groups)
    # print(f"gpu_num:{gpu_num}")
    return opt_comm_op, comm_groups, search_time

def print_result(result, comm_groups):
    for index, r in enumerate(result):
        print(f"{index}th seq: [len:{len(r)}]")
        for op, group_name, comm_type in r:
            print(f"[ {op.name}, {comm_groups[group_name][comm_type]} ]")
        print("------------------")

# 打印 [带宽延迟] 矩阵
def print_matrix(matrix):
    for row in matrix:
        print(" ".join(f"{str(item):<15}" for item in row))

# 生成集群下的通信组
def generate_comm_groups(rtsw, nnode_per_rtsw, ngpu_per_node):
    comm_groups = {
        "rtsw": {},
        "host": {}
    }

    if rtsw != 1:
        comm_groups["ctsw"] = {}
        # ctsw下的所有gpu
        comm_groups["ctsw"]["InsideGroup"] = [[i for i in range(ngpu_per_node * nnode_per_rtsw * rtsw)]]
        # rtsw下的并行gpu
        comm_groups["rtsw"]["Parallel_rtsw"] = [[((ngpu_per_node * nnode_per_rtsw * r)) for r in range(rtsw)]]
    # rtsw下的所有gpu
    comm_groups["rtsw"]["InsideGroup"] = [[i + (r * ngpu_per_node * nnode_per_rtsw) for i in range(ngpu_per_node * nnode_per_rtsw)] for r in range(rtsw)]
    # host下的所有gpu
    comm_groups["host"]["InsideGroup"] = [[i + ngpu_per_node * n  + ngpu_per_node * nnode_per_rtsw * r for i in range(ngpu_per_node)] for n in range(nnode_per_rtsw) for r in range(rtsw)]
    # host下的并行gpu
    comm_groups["host"]["Parallel_host_0"] = [[((ngpu_per_node * n)) for n in range(nnode_per_rtsw * rtsw)]]
    # comm_groups["host"]["Parallel_host_1"] = [[((ngpu_per_node * n + g)) for n in range(nnode_per_rtsw * rtsw)] for g in range(ngpu_per_node)]

    print(f"comm_groups:{comm_groups}")

    return comm_groups

# [已弃用] 生成单交换机下的通信组
def generate_comm_groups_single(n, m): #  n机m卡
    comm_groups = {
        "rack": {
            "InsideGroup": [[i for i in range(n * m)]]
        },
        "server": {}
    }

    # InsideGroup
    comm_groups["server"]["InsideGroup_0"] = [
        [j for j in range(i * m, (i + 1) * m)]
        for i in range(n)
    ]

    comm_groups["server"][f"Parallel_rack_0"] = [[j for j in range(i, n * m, m)] for i in range(m)]

    # Parallel racks
    for i in range(m):
        comm_groups["server"][f"Parallel_rack_{i+1}"] = [[j for j in range(i, n * m, m)]]

    return comm_groups

# 计算集合通信序列的成本
def comm_list_cost(matrix_bandwidth_delay, comm_list, comm_groups, epoch_num):
    list_cost = 0
    max_cost = 0
    for comm in comm_list:
        collective, group_name, comm_type = comm
        group = comm_groups[group_name][comm_type]
        cost_ = get_comm_op_cost(matrix_bandwidth_delay, group[0], collective)
        max_cost = max(max_cost, cost_)
        list_cost += cost_
        # print(f"collective: {collective}, cost: {cost_}")
        
    return list_cost + epoch_num * max_cost

# 计算集合通信算子的成本
def get_comm_op_cost(matrix_bandwidth_delay, group, collective):
    s = 2 / 1024 # 数据量假定为 2MB
    comm_op_cost = 0 # 通信成本
    
    # g = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]], group = [0, 1, 2, 3]
    # g = [[0, 1, 2 ,3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]], group = [0, 1, 2 ,3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    
    if collective == Collective.AllReduce:
        # TODO 计算 AllReduce Cost
        # AllReduce = ReduceScatter + AllGather (ReduceScatter = AllGather)
        comm_group_rank_num = len(group)
        # 通信数据量
        s = s / comm_group_rank_num
        # 找到两两gpu之间的最大通信成本
        for i in range(comm_group_rank_num):
            for j in range(comm_group_rank_num):
                if i == j:
                    continue
                src = group[i]
                dst = group[j]
                bandwidth = matrix_bandwidth_delay[src][dst][0]
                delay = matrix_bandwidth_delay[src][dst][1]
                comm_op_cost = max(comm_op_cost, cost(s, bandwidth, delay))
        return comm_op_cost * 2
    elif collective == Collective.AllGather:
        # TODO 计算 AllGather Cost
        comm_group_rank_num = len(group)
        # 通信数据量
        s = s / comm_group_rank_num
        # 找到两两gpu之间的最大通信成本
        for i in range(comm_group_rank_num):
            for j in range(comm_group_rank_num):
                if i == j:
                    continue
                src = group[i]
                dst = group[j]
                bandwidth = matrix_bandwidth_delay[src][dst][0]
                delay = matrix_bandwidth_delay[src][dst][1]
                comm_op_cost = max(comm_op_cost, cost(s, bandwidth, delay))
        return comm_op_cost
    elif collective == Collective.ReduceScatter:
        # TODO 计算 ReduceScatter Cost (AllGather逆)
        comm_group_rank_num = len(group)
        # 通信数据量
        s = s / comm_group_rank_num
        # 找到两两gpu之间的最大通信成本
        for i in range(comm_group_rank_num):
            for j in range(comm_group_rank_num):
                if i == j:
                    continue
                src = group[i]
                dst = group[j]
                bandwidth = matrix_bandwidth_delay[src][dst][0]
                delay = matrix_bandwidth_delay[src][dst][1]
                comm_op_cost = max(comm_op_cost, cost(s, bandwidth, delay))
        return comm_op_cost
    elif collective == Collective.Broadcast:
        # TODO 计算 Broadcast Cost
        root = 0 # 根节点
        comm_group_rank_num = len(group)
        # 通信数据量
        s = s / comm_group_rank_num
        # 找到从root到其他gpu的最大通信成本
        for i in range(len(group)):
            if i == root:
                continue
            src = group[root]
            dst = group[i]
            bandwidth = matrix_bandwidth_delay[src][dst][0]
            delay = matrix_bandwidth_delay[src][dst][1]
            comm_op_cost = max(comm_op_cost, comm_group_rank_num * cost(s, bandwidth, delay))
        return comm_op_cost
    
    return float('inf')

# [α-β] cost model
def cost(s, bandwidth, delay):
    alpha = delay
    beta = (10**6) / bandwidth
    return alpha + beta * s # 单位为us

if __name__ == "__main__":
    search()