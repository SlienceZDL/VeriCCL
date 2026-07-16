# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from collections import defaultdict

import math

def _distances(topology):
    # Floyd–Warshall algorithm for all-pairs shortest paths with path information
    # Modified to track all shortest paths
    # next 中记录了多条等长的最短路径的第一个 next节点
    nodes = range(topology.num_nodes())
    dist = [[math.inf for _ in nodes] for _ in nodes]
    next = [[set() for _ in nodes] for _ in nodes]
    for dst in nodes:
        for src in topology.sources(dst):
            dist[src][dst] = 1
            next[src][dst].add(dst)
    for node in nodes:
        dist[node][node] = 0
        next[node][node].add(node)
    for k in nodes:
        for i in nodes:
            for j in nodes:
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next[i][j] = set()
                    for l in next[i][k]:
                        next[i][j].add(l)
                elif dist[i][j] == dist[i][k] + dist[k][j]:
                    for l in next[i][k]:
                        next[i][j].add(l)

    return dist, next

def shortest_path_sets(topology, collective):
    """
    计算每个chunk的最短路径集合
    
    Args:
        topology: 网络拓扑结构对象,包含节点间的连接关系和距离信息
        collective: 集合通信对象,包含chunks和它们的前置/后置条件
        
    Returns:
        dict: 一个字典,key为chunk的id,value为该chunk的最短路径集合
              (包含所有可能参与该chunk传输的节点)
    """
    # 计算所有节点对之间的最短距离和下一跳信息
    dist, next = _distances(topology)
    # 初始化最短路径集合字典
    spsets = {}
    
    # 遍历每个chunk
    for id, chunk in enumerate(collective._chunks):
        spset = set()
        for u in chunk.precondition:
            for v in chunk.postcondition:
                curr = next[u][v]
                if not curr:
                    continue
                spset.add(u)
                while not v in curr:
                    spset.update(curr)
                    curr = set().union(*[next[x][v] for x in curr])
                spset.update(curr)
        spsets[id] = spset

    return spsets