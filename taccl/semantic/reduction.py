from taccl.semantic.common_enum import Collective
import numpy as np
import copy

class Reduction:
    """
    Reduction 校验类, 用于按顺序校验 group 中所有 rank 的 reduction 操作是否正确

    Args:
        k (int): rank 数量

    Attributes:
        k (int): rank 数量
        S (list): k个状态列表,S[x]表示第x个rank的状态,每个状态列表中初始包含rank各自的k个分片(即数据),
                每个分片是一个集合(用于存储reduce sum的中间结果),
                集合中的元素是元组(i, j):表示第i个rank的第j个分片.
    """

    def __init__(self, ranks):  # k 个 rank
        self.k = ranks
        self.S = [list() for _ in range(ranks)]  # k个状态列表
        self.init_S()

    def init_S(self):
        for i in range(self.k):  # 第 i 个 rank
            self.S[i] = []
            for j in range(self.k):  # 第 j 个 chunk
                self.S[i].append({(i, j)})

    def print_S(self):
        print("-----------------------------")
        for i in range(self.k):  # 第 i 个 rank
            print(f"rank {i} :\n")
            for j in range(len(self.S[i])):
                print(f"        {self.S[i][j]}\n")
        print("-----------------------------")

    def allreduce(self, group):
        # 检验条件
        g_0 = group[0]

        S0_len = len(self.S[g_0])

        # 1.保证参与allreduce的rank的数据量相同
        for i in group:
            if i == g_0:
                continue
            Si_len = len(self.S[i])
            if S0_len != Si_len:  # chunk数量相等
                return False

        # 2.保证rank中的数据不会被重复reduce sum
        for i in group:
            for j in group:
                if i == j:
                    continue
                for k in range(S0_len):
                    if self.S[i][k].isdisjoint(self.S[j][k]) == False:  # 不再reduce相同rank的相同chunk
                        return False

        # 更新状态 allreduce = reduce + broadcast
        # Reduce, root = g_0
        for i in group:
            if i == g_0:
                continue
            for k in range(S0_len):
                self.S[g_0][k].update(self.S[i][k])

        # Broadcast, root = g_0
        for i in group:
            if i == g_0:
                continue
            self.S[i] = copy.deepcopy(self.S[g_0])
        
        return True

    def reduce(self, group):
        # 检验条件
        g_0 = group[0]

        S0_len = len(self.S[g_0])

        # 1.保证参与reduce的rank的数据量相同
        for i in group:
            if i == g_0:
                continue
            Si_len = len(self.S[i])
            if S0_len != Si_len:  # chunk数量相等
                return False

        # 2.保证rank中的数据不会被重复reduce sum
        for i in group:
            for j in group:
                if i == j:
                    continue
                for k in range(S0_len):
                    if self.S[i][k].isdisjoint(self.S[j][k]) == False:  # 不再reduce相同rank的相同chunk
                        return False

        # 更新状态 reduce到第一个rank
        # Reduce, root = g_0
        for i in group:
            if i == g_0:
                continue
            for k in range(S0_len):
                self.S[g_0][k].update(self.S[i][k])

        for i in group:
            if i == g_0:  # 保留root数据
                continue
            self.S[i].clear()  # 清空其余rank数据

        return True

    def allgather(self, group):
        # 检验条件
        g_0 = group[0]

        S0_len = len(self.S[g_0])

        # 1.保证参与allgather的rank的数据量相同
        for i in group:
            if i == g_0:
                continue
            Si_len = len(self.S[i])
            if S0_len != Si_len:  # chunk数量相等
                return False

        # 更新状态
        # 先gather到g_0
        for i in group:
            if i == g_0:
                continue
            for k in range(S0_len):
                self.S[g_0].append(self.S[i][k])

        # 从 S[g_0] 广播到其他rank
        for i in group:
            if i == g_0:
                continue
            self.S[i] = copy.deepcopy(self.S[g_0])

        return True
    
    def gather(self, group):
        # 检验条件
        g_0 = group[0]

        S0_len = len(self.S[g_0])

        # 1.保证参与gather的rank的数据量相同
        for i in group:
            if i == g_0:
                continue
            Si_len = len(self.S[i])
            if S0_len != Si_len:  # chunk数量相等
                return False

        # 更新状态
        # 先gather到g_0
        for i in group:
            if i == g_0:
                continue
            for k in range(S0_len):
                self.S[g_0].append(self.S[i][k])

        # 清除其他rank的数据
        for i in group:
            if i == g_0:
                continue
            self.S[i].clear()

        return True

    def bostcast(self, group):
        # 检验条件
        g_0 = group[0]

        S0_len = len(self.S[g_0])

        if S0_len == 0:
            return False

        # 1.保证参与bostcast的root的数据量不能小于其他rank(无意义)
        for i in group:
            if i == g_0:
                continue
            Si_len = len(self.S[i])
            if S0_len < Si_len:  # root chunk数量不能小于其它rank
                return False

        '''
        # 2.保证bostcast后其他rank的信息量增加(有意义) (此处费时)
        flag = False
        for i in group:
            if i == g_0:
                continue
            S_other_len = len(self.S[i])
            if S_other_len == 0:
                flag = True
            for k in range(S_other_len):
                if self.S[g_0][k] > self.S[i][k]: # 存在信息增加
                    flag = True
                elif self.S[g_0][k] < self.S[i][k]: # 出现信息量减少，应该排除
                    return False
        if flag == False:
            return False
        '''

        # 更新状态
        for i in group:
            if i == g_0:
                continue
            self.S[i] = copy.deepcopy(self.S[g_0])

        return True

    def reduce_scatter(self, group):
        # 检验条件
        g_len = len(group)
        g_0 = group[0]

        S0_len = len(self.S[g_0])

        # 1.保证Scatter
        if (S0_len % g_len) != 0:  # 能够整除，保证chunk能够均匀分配到所有rank
            return False

        # 2.保证Reduce
        for i in group:
            if i == g_0:
                continue
            Si_len = len(self.S[i])
            if S0_len != Si_len:  # chunk数量相等
                return False
            
        # 2.保证Reduce
        for i in group:
            for j in group:
                if i == j:
                    continue
                for k in range(S0_len):
                    if self.S[i][k].isdisjoint(self.S[j][k]) == False:  # 不再reduce相同rank的相同chunk
                        return False

        # 更新状态
        # Reduce, root = g_0
        for i in group:
            if i == g_0:
                continue
            for k in range(S0_len):
                self.S[g_0][k].update(self.S[i][k])

        # Scatter
        size =  S0_len // g_len
        for i in group:
            if i == g_0:
                continue
            self.S[i].clear()
            for k in range(size):
                self.S[i].append(self.S[g_0].pop(size))

        return True

    def check_and_update(self, op, group):
        if op == Collective.AllReduce:
            return self.allreduce(group)
        elif op == Collective.AllGather:
            return self.allgather(group)
        elif op == Collective.Broadcast:
            return self.bostcast(group)
        elif op == Collective.ReduceScatter:
            return self.reduce_scatter(group)
        elif op == Collective.Reduce:
            return self.reduce(group)
        elif op == Collective.Gather:
            return self.gather
        return False

    def check_allreduce(self):
        # print("---------------------------")
        for i in range(self.k):
            if len(self.S[i]) != self.k:
                return False
            for j in range(self.k):
                if len(self.S[i][j]) != self.k:
                    return False
        return True
    
    def check_allgather(self):
        print("---------------------------")
        len_a = self.k * self.k
        for i in range(self.k):
            if len(self.S[i]) != len_a:
                print(1)
                return False
            set_tmp = set()
            for j in range(len_a):
                if len(self.S[i][j]) != 1:
                    print(2)
                    return False
                # print(f"self.S[i][j] : {self.S[i][j]}\n")
                set_tmp.update(self.S[i][j])
            if len(set_tmp) != len_a:
                print(f"rank : {i+1} ,set_tmp : {set_tmp}\n")
                return False
                
        return True
        
    
# if __name__ == "__main__":
#     r = Reduction(4)

#     r.print_S()
#     r.gather([0,1,2,3])
#     # r.gather([2,3])
#     # r.gather([0,2])
#     r.bostcast([0,1,2,3])

#     r.print_S()
#     # print(r.check_allreduce())
#     print(r.check_allgather())