
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp


class LightGCN(nn.Module):
    '''
    interaction_matrix: coo_matrix
    '''

    def __init__(self, device, n_layers, n_users, n_items, interaction_matrix):
        super(LightGCN, self).__init__()
        self.device = device
        self.n_layers = n_layers
        self.user_count = n_users
        self.item_count = n_items
        self.interaction_matrix = interaction_matrix
        self.A_adj_matrix = self._get_a_adj_matrix()


    def _get_a_adj_matrix(self):

        n = self.user_count + self.item_count
        inter_matrix = self.interaction_matrix.tocoo()

        rows = np.concatenate([inter_matrix.row, inter_matrix.col + self.user_count])
        cols = np.concatenate([inter_matrix.col + self.user_count, inter_matrix.row])
        A = sp.coo_matrix(
            (np.ones_like(rows, dtype=np.float32), (rows, cols)),
            shape=(n, n)
        )

        node_degree = np.array((A > 0).sum(axis=1)).flatten()
        diag = sp.diags(np.power(node_degree + 1e-7, -0.5))
        A_adj = diag @ A @ diag

        A_adj = sp.coo_matrix(A_adj)
        index = torch.from_numpy(np.vstack([A_adj.row, A_adj.col])).long()
        A_sparse = torch.sparse_coo_tensor(
            index,
            torch.FloatTensor(A_adj.data),
            torch.Size(A_adj.shape),
            dtype=torch.float32
        )

        return A_sparse

    def forward(self, in_embs):

        result = [in_embs]
        for i in range(self.n_layers):
            in_embs = torch.sparse.mm(self.A_adj_matrix.to(self.device), in_embs)
            in_embs = F.normalize(in_embs, dim=-1)
            result.append(in_embs / (i + 1))
            # result.append(in_embs)

        result = torch.stack(result, dim=0)
        result = torch.sum(result, dim=0)

        return result


class GraphDistillationLayer(nn.Module):
    def __init__(self, dim, layer_num):
        super().__init__()
        self.dim = dim
        self.layer_num = layer_num

        self.phi = nn.ModuleList([
            nn.Linear(2 * dim, dim, bias=True)
            for _ in range(layer_num)
        ])

    def forward(self, E_p, E_n, mask):
        num_u, seq_len, dim = E_n.shape
        E_updated = E_p.clone()
        mask = (~mask).float()

        for l in range(self.layer_num):
            E_p_expanded = E_updated.unsqueeze(1).expand(-1, seq_len, -1)
            concat = torch.cat([E_p_expanded, E_n], dim=-1)
            similarity = F.relu(self.phi[l](concat))

            neighbor_count = mask.sum(dim=1, keepdim=True).clamp(min=1)
            aggregated = (similarity * mask.unsqueeze(-1)).sum(dim=1) / neighbor_count

            E_updated = E_updated - aggregated

        return E_updated
