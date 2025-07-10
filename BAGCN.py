import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


class BAGCN(nn.Module):
    '''
    interaction_matrix: coo_matrix
    '''

    def __init__(self, device, n_layers, n_users, n_items, embedding_size, interaction_matrix):
        super(BAGCN, self).__init__()
        self.device = device
        self.n_layers = n_layers
        self.user_count = n_users
        self.item_count = n_items
        self.interaction_matrix = interaction_matrix
        self.A_adj_matrix = self._get_a_adj_matrix()

        # 使用 nn.ModuleList 存储每一层的 Linear 层
        self.node_linears = nn.ModuleList()
        self.rel_linears = nn.ModuleList()

        for _ in range(n_layers):
            self.node_linears.append(nn.Linear(embedding_size, embedding_size, bias=False))
            self.rel_linears.append(nn.Linear(embedding_size, embedding_size, bias=False))


    def _get_a_adj_matrix(self):
        """构建归一化的邻接矩阵A~

        Returns:
            torch.sparse.FloatTensor: 归一化的稀疏邻接矩阵
        """
        n = self.user_count + self.item_count
        inter_matrix = self.interaction_matrix.tocoo()

        # 构建对称邻接矩阵
        rows = np.concatenate([inter_matrix.row, inter_matrix.col + self.user_count])
        cols = np.concatenate([inter_matrix.col + self.user_count, inter_matrix.row])
        A = sp.coo_matrix(
            (np.ones(2 * inter_matrix.nnz, dtype=np.float32), (rows, cols)),
            shape=(n, n)
        )

        degrees = np.array((A > 0).sum(axis=1)).flatten() + 1e-7
        D = sp.diags(np.power(degrees, -0.5))
        A_adj = D @ A @ D

        A_adj = A_adj.tocoo()
        indices = torch.from_numpy(np.vstack([A_adj.row, A_adj.col])).long()
        values = torch.FloatTensor(A_adj.data)

        return torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=A_adj.shape,
            dtype=torch.float32,
            device=values.device
        )

    def forward(self, in_embs, beh_embs):


        # beh_embs = beh_embs.squeeze(dim=1)
        result = [in_embs]
        behaviors = [beh_embs]
        for i in range(self.n_layers):
            in_embs = in_embs + beh_embs
            in_embs = self.node_linears[i](in_embs)

            in_embs = torch.sparse.mm(self.A_adj_matrix.to(self.device), in_embs)
            in_embs = F.normalize(in_embs, dim=-1)
            result.append(in_embs / (i + 1))

            beh_embs = self.rel_linears[i](beh_embs)
            behaviors.append(beh_embs / (i + 1))
            # result.append(in_embs)

        result = torch.stack(result, dim=0)
        result = torch.sum(result, dim=0)

        behaviors = torch.stack(behaviors, dim=0)
        behaviors = torch.sum(behaviors, dim=0)


        return result, behaviors