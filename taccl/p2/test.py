if __name__ == "__main__":
    comm_groups = {
        "rtsw": {},
        "host": {}
    }

    ngpu_per_node = 8
    nnode_per_rtsw = 2
    rtsw = 1
    # ctsw下的所有gpu
    if rtsw != 1:
        comm_groups["ctsw"] = {}
        comm_groups["ctsw"]["InsideGroup"] = [[i for i in range(ngpu_per_node * nnode_per_rtsw * rtsw)]]
        comm_groups["rtsw"]["Parallel_rtsw_0"] = [[((ngpu_per_node * nnode_per_rtsw * r)) for r in range(rtsw)]]
        # comm_groups["rtsw"]["Parallel_rtsw_1"] = [[((ngpu_per_node * nnode_per_rtsw * r))] for r in range(rtsw)]
    # rtsw下的所有gpu
    comm_groups["rtsw"]["InsideGroup"] = [[i + (r * ngpu_per_node * nnode_per_rtsw) for i in range(ngpu_per_node * nnode_per_rtsw)] for r in range(rtsw)]
    # rtsw下的并行gpu
    # comm_groups["rtsw"]["Parallel_rtsw"] = [((ngpu_per_node * nnode_per_rtsw * r)) for r in range(rtsw)]
    # host下的所有gpu
    comm_groups["host"]["InsideGroup"] = [[i + ngpu_per_node * n  + ngpu_per_node * nnode_per_rtsw * r for i in range(ngpu_per_node)] for n in range(nnode_per_rtsw) for r in range(rtsw)]
    # host下的并行gpu
    comm_groups["host"]["Parallel_host_0"] = [[((ngpu_per_node * n)) for n in range(nnode_per_rtsw * rtsw)]]
    comm_groups["host"]["Parallel_host_1"] = [[((ngpu_per_node * n + g)) for n in range(nnode_per_rtsw * rtsw)] for g in range(ngpu_per_node)]

    print(comm_groups)


# 定义 comm_groups 和 comm_groups_2 的位置应确保在主进程和子进程中可用
comm_groups = {
    "CPU": {
        "InsideGroup": [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
        "Parallel_server": [[0, 4], [1, 5], [2, 6], [3, 7], [8, 12], [9, 13], [10, 14], [11, 15]],
        "Parallel_rack": [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]],
        "Master_rack": [[0, 4, 8, 12]]
    },
    "server": {
        "InsideGroup": [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
        "Parallel_rack": [[0, 8], [1, 9], [2, 10], [3, 11], [4, 12], [5, 13], [6, 14], [7, 15]]
    },
    "rack": {
        "InsideGroup": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
    }
}

comm_groups_2 = {
    "rack": {
        "InsideGroup": [[0, 1, 2, 3, 4, 5, 6, 7]]
    },
    "server": {
        "InsideGroup_0": [[0, 1, 2, 3], [4, 5, 6 ,7]],
        "Parallel_rack_0": [[0, 4], [1, 5], [2, 6], [3, 7]],
        "Parallel_rack_1": [[0, 4]],
        "Parallel_rack_2": [[1, 5]],
        "Parallel_rack_3": [[2, 6]],
        "Parallel_rack_4": [[3, 7]]
    }
}

comm_groups_2_8 = {
    "rack": {
        "InsideGroup": [[0, 1, 2, 3, 4, 5, 6, 7]]
    },
    "server": {
        "InsideGroup_0": [[0, 1, 2, 3, 4, 5, 6 ,7], [8, 9, 10, 11, 12, 13, 14, 15]], # stage 0 2
        "InsideGroup_1": [[0, 1, 2, 3]],
        "InsideGroup_2": [[4, 5, 6 ,7]],
        "Parallel_rack_0": [[0, 4], [1, 5], [2, 6], [3, 7]],
        "Parallel_rack_1": [[0, 8]], # stage 1
        "Parallel_rack_2": [[1, 5]],
        "Parallel_rack_3": [[2, 6]],
        "Parallel_rack_4": [[3, 7]]
    }
}

comm_groups_3 = {
    "rack": {
        "InsideGroup": [[0, 1, 2, 3]]
    },
    "server": {
        "InsideGroup_0": [[0, 1], [2, 3]],
        "InsideGroup_1": [[0, 1]],
        "InsideGroup_2": [[2, 3]],
        "Parallel_rack_0": [[0, 2], [1, 3]],
        "Parallel_rack_1": [[0, 2]],
        "Parallel_rack_2": [[1, 3]]
    }
}