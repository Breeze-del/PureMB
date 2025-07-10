import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

from data_set import DataSet
from utils import BPRLoss, EmbLoss
from lightGCN import LightGCN,GraphDistillationLayer
from Transformer import LinearTransformer


class PureMB(nn.Module):
    def __init__(self, args, dataset: DataSet):
        super(PureMB, self).__init__()

        self.device = args.device
        self.layers = args.layers
        self.layer_num = args.layer_num
        self.reg_weight = args.reg_weight
        self.log_reg = args.log_reg
        self.node_dropout = args.node_dropout
        self.message_dropout = nn.Dropout(p=args.message_dropout)
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.inter_matrix = dataset.inter_matrix
        self.user_item_inter_set = dataset.user_item_inter_set
        self.test_users = list(dataset.test_interacts.keys())
        self.behaviors = args.behaviors
        self.embedding_size = args.embedding_size
        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)
        self.bhv_embs = nn.Parameter(torch.eye(len(self.behaviors)))
        self.global_Graph = LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, dataset.all_inter_matrix)
        # self.behavior_Graph = LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, dataset.inter_matrix[-1])
        self.behavior_Graph_list = nn.ModuleList()
        for i in range(len(self.behaviors)):
            self.behavior_Graph_list.append(
                LightGCN(self.device, self.layers, self.n_users + 1, self.n_items + 1, dataset.inter_matrix[i]))
        self.transformer = LinearTransformer(
            in_channels=self.embedding_size,
            hidden_channels=self.embedding_size,
            out_channels=self.embedding_size,
            num_layers=1,
            num_heads=1
        )
        self.gate_layer = nn.Linear(self.embedding_size * 2, self.embedding_size, bias=True)

        # cross-attention
        self.W_q = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.W_k = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.W_v = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.out_proj = nn.Linear(self.embedding_size, self.embedding_size, bias=True)  # optional
        self.gate_linear_att = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.layernorm = nn.LayerNorm(self.embedding_size)

        self.pre_dis_layer = GraphDistillationLayer(self.embedding_size, layer_num=self.layer_num)
        self.RZ = nn.Linear(2 * self.embedding_size ,self.embedding_size * 2, bias=True)
        self.U = nn.Linear(2 * self.embedding_size ,self.embedding_size, bias=True)

        self.reg_weight = args.reg_weight
        self.layers = args.layers
        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()
        self.cross_loss = nn.BCELoss()

        self.model_path = args.model_path
        self.check_point = args.check_point
        self.if_load_model = args.if_load_model
        # self.message_dropout = nn.Dropout(p=args.message_dropout)

        self.storage_user_embeddings = None
        self.storage_item_embeddings = None

        # load potential_negative for tmall or taobao dataset
        self.potential_negative = self._potential_negative()

        self.apply(self._init_weights)

        self._load_model()

    def _init_weights(self, module):

        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _load_model(self):
        if self.if_load_model:
            parameters = torch.load(os.path.join(self.model_path, self.check_point))
            self.load_state_dict(parameters, strict=False)

    def _potential_negative(self):
        if self.n_items == 11953:
            file_path = os.path.join("./data/Tmall", 'potential_negative_list.txt')
            with open(file_path, 'r', encoding='utf-8') as f:
                potential_neg_list = json.load(f)
            # self.user_item_inter_set.insert(len(self.user_item_inter_set) - 1, potential_neg_list)
                return potential_neg_list
        else:
            return None

    def agg_info(self, u_emb, i_emb):
        in_feature = torch.cat((u_emb, i_emb), dim=-1)  # [B, L, 2d]
        RZ = torch.sigmoid(self.RZ(in_feature))  # [B, L, 2d]
        R, Z = torch.chunk(RZ, 2, dim=-1)  # [B, L, d], [B, L, d]

        RU = R * u_emb  # reset gate: reset user embedding
        RU_cat = torch.cat((RU, i_emb), dim=-1)  # combine with item
        candidate = torch.tanh(self.U(RU_cat))  # candidate representation

        u_final = (1 - Z) * u_emb + Z * candidate  # update gate: interpolate
        return u_final


    def forward(self, batch_data):
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = self.global_Graph(all_embeddings)
        # user_embedding, item_embedding = torch.split(all_embeddings, [self.n_users + 1, self.n_items + 1])

        user_behavior_embeddings = []
        item_behavior_embeddings = []
        for i in range(len(self.behaviors)):
            node_embedding= self.behavior_Graph_list[i](all_embeddings)
            user_b_embedding, item_b_embedding = torch.split(node_embedding, [self.n_users + 1, self.n_items + 1])
            user_behavior_embeddings.append(user_b_embedding)
            item_behavior_embeddings.append(item_b_embedding)
        # list change to tensor
        user_embeddings = torch.stack(user_behavior_embeddings)  # [4, num_users, emb_dim]
        item_embeddings = torch.stack(item_behavior_embeddings)  # [4, num_items, emb_dim]

        log_loss = self.calculate_cross_loss(batch_data, user_embeddings, item_embeddings)

        pair_samples = batch_data[:, -1, :-1]
        mask = torch.any(pair_samples != 0, dim=-1)
        pair_samples = pair_samples[mask]
        bpr_loss = 0
        if pair_samples.shape[0] > 0:
            user_samples = pair_samples[:, 0].long()
            item_samples = pair_samples[:, 1:].long()
            i_emb_target = item_embeddings[-1][item_samples]

            batcah_user_embeddings = user_embeddings[:, user_samples].permute(1, 0, 2) # [num_users, 4, emb_dim]
            user_bf_embedding = self.attention_aggregate(batcah_user_embeddings)
            neg_pre, neg_mask, neg_preference = self.preference_dis(user_samples, self.user_item_inter_set[-2],
                                                          batcah_user_embeddings[:, -2], item_embeddings[-2])

            # # distillation for taobao or tmall dataset
            # neg_pre, neg_mask, neg_preference = self.preference_dis(user_samples, self.potential_negative,
            #                                                         batcah_user_embeddings[:, -1], item_embeddings[-1])
            user_enhance_embedding = self.pre_dis_layer(user_bf_embedding, neg_pre, neg_mask)
            # tar_enh_embedding = batcah_user_embeddings[:, -1] * 2
            tar_enh_embedding = self.transformer(user_embeddings[-1])[user_samples] +(batcah_user_embeddings[:, -1] *2)
            # _, _, tar_enh_embedding = self.preference_dis(user_samples, self.user_item_inter_set[-1],
            #                                               batcah_user_embeddings[:, -1], item_embeddings[-1])
            gate = torch.sigmoid(self.gate_layer(torch.cat([user_enhance_embedding, tar_enh_embedding], dim=-1)))  # [batch_size, emb_dim]
            final_u = (gate * user_enhance_embedding + (1 - gate) * tar_enh_embedding).unsqueeze(1).expand(-1, 2, -1)

            # final_u = F.normalize(final_u, p=2, dim=-1)
            # i_emb_target = F.normalize(i_emb_target, p=2, dim=-1)
            # neg_preference = F.normalize(neg_preference, p=2, dim=-1)

            bpr_scores = torch.sum((final_u * i_emb_target ), dim=-1)
            neg_scores = torch.sum((neg_preference * i_emb_target[:,0]), dim=-1)
            # tar_scores = torch.sum((tar_preference * i_emb_target[:, 0]), dim=-1)

            p_scores, n_scores = torch.chunk(bpr_scores, 2, dim=-1)
            bpr_loss += self.bpr_loss(p_scores, n_scores) + self.bpr_loss(p_scores, neg_scores)
            # bpr_loss += self.PSL_loss(p_scores, n_scores, tau=0.1) + self.PSL_loss(p_scores, neg_scores, tau=0.1)
        emb_loss = self.emb_loss(self.user_embedding.weight, self.item_embedding.weight)
        loss = self.log_reg * log_loss + (1 - self.log_reg) * bpr_loss + self.reg_weight * emb_loss

        return loss

    def full_predict(self, users):
        if self.storage_user_embeddings is None:
            all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
            all_embeddings = self.global_Graph(all_embeddings)

            user_behavior_embeddings = []
            item_behavior_embeddings = []
            for i in range(len(self.behaviors)):
                node_embedding = self.behavior_Graph_list[i](all_embeddings)
                user_b_embedding, item_b_embedding = torch.split(node_embedding, [self.n_users + 1, self.n_items + 1])
                user_behavior_embeddings.append(user_b_embedding)
                item_behavior_embeddings.append(item_b_embedding)
            # list change to tensor
            user_embeddings = torch.stack(user_behavior_embeddings)  # [4, num_users, emb_dim]
            item_embeddings = torch.stack(item_behavior_embeddings)  # [4, num_items, emb_dim]

            self.storage_user_embeddings = torch.zeros(self.n_users + 1, self.embedding_size).to(self.device)

            test_users = [int(x) for x in self.test_users]
            tmp_emb_list = []
            for i in range(0, len(test_users), 100):
                tmp_users = test_users[i: i + 100]
                tmp_users = torch.LongTensor(tmp_users)

                batcah_user_embeddings = user_embeddings[:, tmp_users].permute(1, 0, 2)  # [num_users, 4, emb_dim]
                user_bf_embedding = self.attention_aggregate(batcah_user_embeddings)

                neg_pre, neg_mask, _ = self.preference_dis(tmp_users, self.user_item_inter_set[-2],
                                                                        batcah_user_embeddings[:, -2],
                                                                        item_embeddings[-2])

                # neg_pre, neg_mask, _ = self.preference_dis(tmp_users, self.potential_negative,
                #                                            batcah_user_embeddings[:, -1],
                #                                            item_embeddings[-1])
                user_enhance_embedding = self.pre_dis_layer(user_bf_embedding, neg_pre, neg_mask)
                # tar_enh_embedding = batcah_user_embeddings[:, -1] * 2
                tar_enh_embedding = self.transformer(user_embeddings[-1])[tmp_users] +(batcah_user_embeddings[:, -1] *2)
                # _, _, tar_enh_embedding = self.preference_dis(tmp_users, self.user_item_inter_set[-1],
                #                                               batcah_user_embeddings[:, -1], item_embeddings[-1])
                gate = torch.sigmoid(
                    self.gate_layer(torch.cat([user_enhance_embedding, tar_enh_embedding], dim=-1)))  # [batch_size, emb_dim]
                final_u = (gate * user_enhance_embedding + (1 - gate) * tar_enh_embedding)
                # final_u = user_enhance_embedding + batcah_user_embeddings[:, -1]

                # final_u = F.normalize(final_u, p=2, dim=-1)
                tmp_emb_list.append(final_u)
            tmp_emb_list = torch.cat(tmp_emb_list, dim=0)
            for index, key in enumerate(test_users):
                self.storage_user_embeddings[key] = tmp_emb_list[index]

            self.storage_item_embeddings = item_embeddings[-1]

        user_emb = self.storage_user_embeddings[users.long()]
        scores = torch.matmul(user_emb, self.storage_item_embeddings.transpose(0, 1))

        return scores

    def calculate_cross_loss(self, batch_data, user_embs, item_embs):

        p_samples = batch_data[:, 0, :]  # positive samples [n, 4]
        n_samples = batch_data[:, 1:-1, :].reshape(-1, 4)  # negative samples [n*(m-1), 4]
        samples = torch.cat([p_samples, n_samples], dim=0)  # combined samples [n + n*(m-1), 4]

        u_sample, i_samples, b_samples, gt_samples = torch.chunk(samples, 4, dim=-1)
        u_sample = u_sample.squeeze(-1).long() # [total_samples]
        i_samples = i_samples.squeeze(-1).long() # [total_samples]
        b_samples = b_samples.squeeze(-1).long()  # [total_samples], 值范围0-3
        gt_samples = gt_samples.squeeze(-1).float() # [total_samples]

        b_samples = b_samples.clamp(0, user_embs.size(0) - 1)
        u_emb_selected = user_embs[b_samples, u_sample]
        i_emb_selected = item_embs[b_samples, i_samples]
        u_final = self.agg_info(u_emb_selected, i_emb_selected)

        scores = torch.sum(u_final * i_emb_selected , dim=-1)  # [total_samples]

        probs = torch.sigmoid(scores)  # [n, 5]

        loss = self.cross_loss(probs, gt_samples)

        return loss

    def attention_aggregate(self, A):
        """
                       A: Tensor [num_users, num_behaviors, emb_dim]
                       """
        Q = A[:, -1:]  # [B, 1, emb_dim]
        K = A[:, :-1]  # [B, n_aux, emb_dim]
        V = K

        Q = self.W_q(Q)  # [B, 1, attn_dim]
        K = self.W_k(K)  # [B, n_aux, attn_dim]
        V = self.W_v(V)  # [B, n_aux, attn_dim]

        scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, 1, n_aux]
        attn_weights = F.softmax(scores / (self.embedding_size ** 0.5), dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # [B, 1, attn_dim]
        attn_output = self.message_dropout(attn_output)

        gate = torch.sigmoid(self.gate_linear_att(Q))  # [B, 1, attn_dim]
        fusion = gate * attn_output + (1 - gate) * Q  # [B, 1, attn_dim]

        out = self.out_proj(fusion)
        out = self.layernorm(out)
        return out.squeeze(1)  # [B, emb_dim]

    def graph_distillation(self, user_samples, user_embeds, item_embeds, user_fusion):
        def preference_distillation(keys, user_item_inter_set, user_embed, item_embed):
            agg_items = [user_item_inter_set[x] for x in keys]
            max_len = max(len(l) for l in agg_items)
            padded_list = np.zeros((len(agg_items), max_len), dtype=int)
            for i, l in enumerate(agg_items):
                padded_list[i, :len(l)] = l
            padded_list = torch.from_numpy(padded_list).to(self.device)
            mask = (padded_list == 0)
            agg_item_emb = item_embed[padded_list.long()]
            u_in = user_embed.unsqueeze(1).expand(-1, max_len, -1)
            u_final = self.agg_info(u_in, agg_item_emb)
            return u_final, mask

        keys = user_samples.tolist()
        # negative preference
        neg_pre, neg_mask = preference_distillation(keys, self.user_item_inter_set[-2], user_embeds[:,-2], item_embeds[-2])
        neg_pref = torch.where(neg_mask.unsqueeze(-1), torch.zeros_like(neg_pre), neg_pre)
        neg_pref = torch.sum(neg_pref, dim=1)

        u_final = self.pre_dis_layer(user_fusion, neg_pre, neg_mask)

        return u_final, neg_pref

    def preference_dis(self, user_samples, user_item_inter_set, user_embeds, item_embeds):
        def preference_distillation(keys, user_item_inter_set, user_embed, item_embed):
            agg_items = [user_item_inter_set[x] for x in keys]
            max_len = max(len(l) for l in agg_items)
            padded_list = np.zeros((len(agg_items), max_len), dtype=int)
            for i, l in enumerate(agg_items):
                padded_list[i, :len(l)] = l
            padded_list = torch.from_numpy(padded_list).to(self.device)
            mask = (padded_list == 0)
            agg_item_emb = item_embed[padded_list.long()]
            u_in = user_embed.unsqueeze(1).expand(-1, max_len, -1)
            u_final = self.agg_info(u_in, agg_item_emb)
            return u_final, mask

        keys = user_samples.tolist()

        # target preference
        pre_list, mask = preference_distillation(keys, user_item_inter_set, user_embeds,
                                                    item_embeds)
        pref = torch.where(mask.unsqueeze(-1), torch.zeros_like(pre_list), pre_list)
        pref = torch.sum(pref, dim=1)
        return pre_list, mask, pref

    # alternative main function-PSL
    def PSL_loss(self, p_scores, n_scores, tau=1.0, activation='relu'):
        # τ ∈ {0.005, 0.025, 0.05, 0.1, 0.25}

        if activation == 'tanh':
            sigma = lambda x: torch.log(torch.tanh(x) + 1)
        elif activation == 'relu':
            sigma = lambda x: torch.log(torch.relu(x + 1))
        elif activation == 'atan':
            sigma = lambda x: torch.log(torch.atan(x) + 1)
        else:
            raise ValueError(
                f"Invalid activation function for PSL: {activation}, must be one of ['tanh', 'relu', 'atan']")

        d = (n_scores - p_scores)/2  # (B, N)
        loss = torch.exp(sigma(d) / tau).mean()
        loss = torch.log(loss)
        return loss

