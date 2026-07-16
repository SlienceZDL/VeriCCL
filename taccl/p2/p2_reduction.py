from taccl.p2.p2_enum import Collective
import numpy as np
import copy

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
        self.ranks = ranks
        self.matrix = np.zeros((ranks, ranks, ranks), dtype=np.int8)
        self.init_matrix()

    def init_matrix(self):
        ranks = self.ranks
        for x in range(ranks):
            self.matrix[x, :, :] = 0
            self.matrix[x, :, x] = 1

    def print_matrix(self):
        ranks = self.ranks
        print("************************")
        for r in range(ranks):
            print(f"rank {r}:")
            for c in range(ranks):
                print(self.matrix[r, c])
            print("--------------------------------")
        print("************************")

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

        # 条件1: 每个rank的矩阵的非零行的位置相等
        first_nonzero_rows = np.any(self.matrix[group[0]] != 0, axis=1)
        for i in group[1:]:
            current_nonzero_rows = np.any(self.matrix[i] != 0, axis=1)
            if not np.array_equal(first_nonzero_rows, current_nonzero_rows):
                return False
        
        # 条件2: 矩阵不相交
        for idx, i in enumerate(group):
            for j in group[idx+1:]:
                if np.any(np.logical_and(self.matrix[i] > 0, self.matrix[j] > 0)):
                    return False
                
        # 更新矩阵
        root = group[0]
        # 先 Reduce 到 root rank
        for i in group[1:]:
            self.matrix[root] = self.matrix[root] ^ self.matrix[i]
        # 再 broadcast 到所有rank
        for i in group[1:]:
            self.matrix[i] = self.matrix[root]

        return True
    
    def reduce(self, group):

        # 条件1: 每个rank的矩阵的非零行的位置相等
        first_nonzero_rows = np.any(self.matrix[group[0]] != 0, axis=1)
        for i in group[1:]:
            current_nonzero_rows = np.any(self.matrix[i] != 0, axis=1)
            if not np.array_equal(first_nonzero_rows, current_nonzero_rows):
                return False
        
        # 条件2: 矩阵不相交
        for idx, i in enumerate(group):
            for j in group[idx+1:]:
                if np.any(np.logical_and(self.matrix[i] > 0, self.matrix[j] > 0)):
                    return False
                    
        # 更新矩阵
        root = group[0]
        # Reduce 到 root rank
        for i in group[1:]:
            self.matrix[root] = self.matrix[root] ^ self.matrix[i]
        # 清空其他rank的矩阵
        for i in group[1:]:
            self.matrix[i] = np.zeros((self.ranks, self.ranks), dtype=np.int8)

        return True
    
    def reduce_scatter(self, group):

        # 条件1: 每个rank的矩阵的非零行的位置相等
        first_nonzero_rows = np.any(self.matrix[group[0]] != 0, axis=1)
        for i in group[1:]:
            current_nonzero_rows = np.any(self.matrix[i] != 0, axis=1)
            if not np.array_equal(first_nonzero_rows, current_nonzero_rows):
                return False
            
        # 获取非0行的数量
        nonzero_rows = np.any(self.matrix[group[0]] != 0, axis=1)
        nonzero_rows_count = np.sum(nonzero_rows)
        # 条件2: 非0行能够整除group的数量
        if nonzero_rows_count % len(group) != 0:
            return False
        
        # 条件3: 矩阵不相交
        for idx, i in enumerate(group):
            for j in group[idx+1:]:
                if np.any(np.logical_and(self.matrix[i] > 0, self.matrix[j] > 0)):
                    return False
                
        # 更新矩阵
        # 先 Reduce 到 临时矩阵中
        temp_matrix = copy.deepcopy(self.matrix[group[0]])
        for i in group[1:]:
            temp_matrix = temp_matrix ^ self.matrix[i]

        # 清空所有矩阵
        for i in group:
            self.matrix[i] = np.zeros((self.ranks, self.ranks), dtype=np.int8)
        
        step = nonzero_rows_count // len(group)
        # 再 Scatter 到所有rank
        for idx, i in enumerate(group):
            for c_g_s in range(step):
                for r in range(self.ranks): # 行
                    if nonzero_rows[r] == True: # 这一行非0
                        self.matrix[i][r] = temp_matrix[r]
                        nonzero_rows[r] = False
                        break

        return True

    def allgether(self, group):
        
        # 条件1: 矩阵的非0行的位置不相交
        # 将每个rank的非0行的位置记录下来
        nonzero_rows_positions = {}
        for idx, i in enumerate(group):
            # 获取当前rank的非0行
            current_nonzero_rows = np.where(np.any(self.matrix[i] != 0, axis=1))[0]
            nonzero_rows_positions[i] = current_nonzero_rows
            
            for j in group[idx+1:]:
                # 获取其他rank的非0行
                other_nonzero_rows = np.where(np.any(self.matrix[j] != 0, axis=1))[0]
                # 检查两个rank的非0行是否有重叠
                if len(np.intersect1d(current_nonzero_rows, other_nonzero_rows)) > 0:
                    return False
                
        # 获取非0行的数量
        nonzero_rows = np.any(self.matrix[group[0]] != 0, axis=1)
        nonzero_rows_count = np.sum(nonzero_rows)
        # 条件2: 非0行的数量相等
        for i in group[1:]:
            if np.sum(np.any(self.matrix[i] != 0, axis=1)) != nonzero_rows_count:
                return False
        
        step = nonzero_rows_count // len(group)

        # 更新矩阵
        # 先Gather到root的临时矩阵
        temp_matrix = np.zeros((self.ranks, self.ranks), dtype=np.int8)
        
        # 遍历每个rank，将其非零行合并到临时矩阵中
        for i in group:
            for row_idx in nonzero_rows_positions[i]:
                temp_matrix[row_idx] = self.matrix[i][row_idx]
                
        # 再broadcast到所有rank
        for i in group:
            self.matrix[i] = temp_matrix.copy()

        return True

    def bostcast(self, group):
        # # 条件1: 存在一个矩阵<root矩阵
        root = group[0]
        root_rows = np.any(self.matrix[root] != 0, axis=1)
        # root_nonzero_count = np.sum(root_rows)
        
        # # 检查是否至少有一个其他rank的矩阵的非零元素数量小于root
        # found_less = False
        # for i in group[1:]:
        #     current_rows = np.any(self.matrix[i] != 0, axis=1)
        #     current_nonzero_count = np.sum(current_rows)
        #     if current_nonzero_count < root_nonzero_count:
        #         found_less = True
        #         break
        
        # if not found_less:
        #     print("条件1不满足")
        #     return False

        # 条件2: 任意矩阵<=root矩阵
        # 检查每个rank的矩阵中，如果存在root矩阵没有的非零元素，则不符合条件
        for i in group[1:]:
            # 对于任何root为0的位置，其他矩阵也必须为0
            if np.any(np.logical_and(self.matrix[i] != 0, np.logical_not(root_rows.reshape(-1, 1)))):
                return False
        
        # 更新矩阵
        # 将root矩阵Broadcast到所有rank
        for i in group[1:]:
            self.matrix[i] = self.matrix[root].copy()

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
        elif op == Collective.Reduce:
            return self.reduce(group)
        else:
            print("不支持的操作")
        return False

    def check_allreduce(self):
        return np.all(self.matrix == 1)
    
if __name__ == "__main__":
    reduction = Reduction(8)
    reduction.print_matrix()
    if reduction.check_and_update(Collective.AllReduce, [0, 1, 2, 3]):
        reduction.print_matrix()
    else:
        print("AllReduce 操作失败")

    if reduction.check_and_update(Collective.Reduce, [4, 5, 6, 7]):
        reduction.print_matrix()
    else:
        print("Reduce 操作失败")

    if reduction.check_and_update(Collective.AllReduce, [0, 4]):
        reduction.print_matrix()
    else:
        print("AllReduce 操作失败")
    
    if reduction.check_and_update(Collective.Broadcast, [0, 1, 2, 3]):
        reduction.print_matrix()
    else:
        print("Broadcast 操作失败")

    if reduction.check_and_update(Collective.Broadcast, [4, 5, 6, 7]):
        reduction.print_matrix()
    else:
        print("Broadcast 操作失败")

    print(reduction.check_allreduce())