from taccl.p2.p2_enum import Collective
import numpy as np

class Reduction:
    """
    Reduction 校验类, 用于按顺序校验 group 中所有 rank 的 reduction 操作是否正确

    Args:
        k (int): rank 数量

    Attributes:
        k (int): rank 数量
        G (numpy.ndarray): k个状态矩阵表示每个rank的从其他rank接收数据的中间状态,每个状态矩阵有k行k列, 
        G[x][i][j]表示第x个rank的第i个分片是否从第j个rank接收到数据,收到为1, 否则为0
    """
    def __init__(self, ranks):
        """
        Args:
            ranks (int): rank 数量
        """
        self.k = ranks
        self.G = np.zeros((ranks, ranks, ranks))
        self.init_G()

    def init_G(self):
        ranks = self.k
        for x in range(ranks):
            # 初始化状态矩阵, 第x个rank的状态矩阵的第x列初始状态为1,其余为0
            self.G[x, :, :] = 0
            self.G[x, :, x] = 1

    def allreduce(self, group):
        '''
        先判断group中的所有rank的状态矩阵是否能够进行allreduce
        如果可以则进行allreduce操作,更新状态矩阵,返回True
        否则返回False

        Args:
            group (list): 参与 allreduce 操作的 rank 列表

        Returns:
            bool: 如果group中的所有rank的状态矩阵能够进行allreduce,则返回True,否则返回False
        '''
        k = self.k
        nonzero_counts = [np.sum(np.any(self.G[i] != 0, axis=1)) for i in group]
        if not all(count == nonzero_counts[0] for count in nonzero_counts):
            return False
        for i in group:
            for j in group:
                if i != j and not np.all((self.G[i] * self.G[j]) == 0):
                    return False
        for i in group:
            ones_positions = np.where(self.G[i] == 1)
            for j in group:
                if i != j:
                    self.G[j][ones_positions] = 1
        return True
    
    def reduce_scatter(self, group):
        k = self.k
        nonzero_counts = [np.sum(np.any(self.G[i] != 0, axis=1)) for i in group]
        if not all(count == nonzero_counts[0] for count in nonzero_counts):
            return False
        for i in group:
            for j in group:
                if i != j and not np.all((self.G[i] * self.G[j]) == 0):
                    return False
        # 修改值
        size = self.k // len(group)
        for i in group:
            ones_positions = np.where(self.G[i] == 1)
            for j in group:
                if i != j:
                    self.G[j][ones_positions] = 1

        for i in group:
            for m in range(self.k):
                if m not in range(size * group.index(i), size * (group.index(i) + 1)):
                    for n in group:
                        self.G[i][m][n] = 0
                        
        return True

    def allgether(self, group):
        k = self.k
        nonzero_counts = [np.sum(np.any(self.G[i] != 0, axis=1)) for i in group]
        ones_counts = [np.sum(self.G[i] == 1) for i in group]
        if not (all(count == nonzero_counts[0] for count in nonzero_counts) and
                all(count == ones_counts[0] for count in ones_counts)):
            return False
        for i in group:
            for j in group:
                if i != j and not np.all((self.G[i] * self.G[j]) == 0):
                    return False
        for i in group:
            for j in group:
                if i != j:
                    self.G[j, i] = np.where(self.G[i, i] == 1, 1, self.G[j, i])
        return True

    def bostcast(self, group):
        k = self.k
        i_0 = group[0]
        Gi_0 = self.G[i_0]
        count_Gi_0_ones = np.sum(Gi_0 == 1)
        G0_zero_mask = (Gi_0 == 0)
        flag = False
        for i in group:
            if i == i_0:
                continue
            if not np.all(self.G[i][G0_zero_mask] == 0):
                return False
            Gi_ones_count = np.sum(self.G[i] == 1)
            if count_Gi_0_ones > Gi_ones_count:
                flag = True
        if not flag:
            return False
        for i in group:
            if i != i_0:
                self.G[i] = self.G[i_0].copy()
        return True

    def check_and_update(self, op, group):
        if op == Collective.AllReduce:
            return self.allreduce(group)
        elif op == Collective.AllGather:
            return self.allgether(group)
        elif op == Collective.Broadcast:
            return self.bostcast(group)
        elif op == Collective.ReduceScatter:
            return self.reduce_scatter(group)
        return False

    def check_full(self):
        return np.all(self.G == 1)
    
if __name__ == "__main__":
    reduction = Reduction(4)
    reduction.print_G()