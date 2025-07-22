import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil


class DynamicChunkEncoder(nn.Module):
    def __init__(self, embed_dim=64, num_heads=8, ffn_dim=256, dropout=0.1, min_chunk=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.min_chunk = min_chunk  

        
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_out = nn.Linear(embed_dim, embed_dim)

        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim)
        )

        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def dynamic_chunking(self, seq_len):
        return max(self.min_chunk, ceil(seq_len / 4))

    def masked_attention(self, q, k, v, mask):
        attn_mask = mask.bool()  # [B,L] -> [B,L] (True=padding)

        B, L, _ = q.shape
        chunk_size = self.dynamic_chunking(L)
        num_chunks = ceil(L / chunk_size)

        if L % chunk_size != 0:
            pad_len = num_chunks * chunk_size - L
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            attn_mask = F.pad(attn_mask, (0, pad_len), value=True)

        # [B, num_chunks, chunk_size, D]
        q = q.view(B, num_chunks, chunk_size, -1)
        k = k.view(B, num_chunks, chunk_size, -1)
        v = v.view(B, num_chunks, chunk_size, -1)
        attn_mask = attn_mask.view(B, num_chunks, chunk_size)

        attn_output = torch.zeros_like(q)
        for i in range(num_chunks):
            start = max(0, i - 1)
            end = min(num_chunks, i + 2)
            scores = torch.einsum('bqd,bkd->bqk',
                                  q[:, i],
                                  k[:, start:end].reshape(B, -1, self.head_dim)) / (self.head_dim ** 0.5)

            chunk_mask = attn_mask[:, start:end].reshape(B, -1)
            scores = scores.masked_fill(chunk_mask.unsqueeze(1), float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            attn_output[:, i] = torch.einsum('bqk,bkd->bqd',
                                             attn_weights,
                                             v[:, start:end].reshape(B, -1, self.head_dim))

        return attn_output.reshape(B, num_chunks * chunk_size, -1)[:, :L]

    def forward(self, x, mask):
        """
        Input:
            x: [batch_size, seq_len, embed_dim]
            mask: [batch_size, seq_len] (0=true, 1=padding)
        """
        B, L, D = x.shape
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
                   for t in qkv]  # [B,h,L,d]

        attn_out = self.masked_attention(
            q.transpose(1, 2).reshape(B * self.num_heads, L, self.head_dim),
            k.transpose(1, 2).reshape(B * self.num_heads, L, self.head_dim),
            v.transpose(1, 2).reshape(B * self.num_heads, L, self.head_dim),
            mask.unsqueeze(1).expand(-1, self.num_heads, -1).reshape(B * self.num_heads, L)
        ).reshape(B, L, D)

        x = x + self.dropout(self.attn_out(attn_out))
        x = self.norm1(x)

        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)

        return x


class OptimizedTransformer(nn.Module):
    def __init__(self, num_layers=4, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            DynamicChunkEncoder(**kwargs)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=64, num_heads=8, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim is wrong"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Input:
            x: [batch_size, seq_len, embed_dim]
            mask: [batch_size, seq_len] (0=True, 1=Padding)
        Output:
            [batch_size, seq_len, embed_dim]
        """
        B, L, D = x.shape

        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
                   for t in qkv]  # [B,h,L,d]

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B,h,L,L]

        # attn_mask = mask.unsqueeze(1).unsqueeze(1)  # [B,1,1,L]
        # scores = scores.masked_fill(attn_mask == 1, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)  # [B,h,L,d]
        context = context.transpose(1, 2).reshape(B, L, D)  # [B,L,D]

        return self.out_proj(context)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim=64, num_heads=8, ffn_dim=256, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.self_attn(x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)

        return x


class StandardTransformerEncoder(nn.Module):
    def __init__(self, num_layers=4, embed_dim=64, num_heads=8, ffn_dim=256, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        Input:
            x: [batch_size, seq_len, embed_dim]
            mask: [batch_size, seq_len] (0=True, 1=Padding)
        Output:
            [batch_size, seq_len, embed_dim]
        """
        for layer in self.layers:
            x = layer(x)
        return x


def full_attention_conv(qs, ks, vs, output_attn=False):
    # normalize input
    qs = qs / torch.norm(qs, p=2)  # [N, H, M]
    ks = ks / torch.norm(ks, p=2)  # [L, H, M]
    N = qs.shape[0]

    # numerator
    kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
    attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
    attention_num += N * vs

    # denominator
    all_ones = torch.ones([ks.shape[0]]).to(ks.device)
    ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
    attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

    # attentive aggregated results
    attention_normalizer = torch.unsqueeze(
        attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
    attention_normalizer += torch.ones_like(attention_normalizer) * N
    attn_output = attention_num / attention_normalizer  # [N, H, D]

    # compute attention for visualization if needed
    if output_attn:
        attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1)  # [N, N]
        normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdims=True)  # [N,1]
        attention = attention / normalizer

    if output_attn:
        return attn_output, attention
    else:
        return attn_output


class TransConvLayer(nn.Module):
    '''
    transformer with fast attention
    '''

    def __init__(self, in_channels,
                 out_channels,
                 num_heads,
                 use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight


    def forward(self, query_input, source_input, output_attn=False):
        # feature transformation
        query = self.Wq(query_input).reshape(-1,
                                             self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(-1,
                                            self.num_heads, self.out_channels)
        if self.use_weight:
            value = self.Wv(source_input).reshape(-1,
                                                  self.num_heads, self.out_channels)
        else:
            value = source_input.reshape(-1, 1, self.out_channels)

        # compute full attentive aggregation
        if output_attn:
            attention_output, attn = full_attention_conv(
                query, key, value, output_attn)  # [N, H, D]
        else:
            attention_output = full_attention_conv(
                query, key, value)  # [N, H, D]

        final_output = attention_output
        final_output = final_output.mean(dim=1)

        if output_attn:
            return final_output, attn
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, num_heads=1,
                 alpha=0.5, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=False):
        super().__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                TransConvLayer(hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.residual = use_residual
        self.alpha = alpha
        self.use_act = use_act

    def forward(self, x):

        layer_ = []

        # input MLP layer
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with full attention aggregation
            x = conv(x, x)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]


class LinearTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, num_heads=1,
                 alpha=0.5, dropout=0.5, use_bn=False, use_residual=True, use_weight=False, use_act=False,
                 graph_weight=0.8):
        super().__init__()
        self.trans_conv = TransConv(in_channels, hidden_channels, num_layers, num_heads, alpha, dropout, use_bn,
                                    use_residual, use_weight)
        self.graph_weight = graph_weight

        self.fc = nn.Linear(hidden_channels, out_channels)


    def forward(self, data):
        x1 = self.fc(self.trans_conv(data))
        return x1

    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x)  # [layer num, N, N]

        return attns



#######################
# text fuction
#######################
if __name__ == "__main__":
    in_dim = 64
    hidden_dim = 64
    num_nodes = 80000
    num_layers = 1
    num_heads = 1

    dummy_input = torch.randn(num_nodes, in_dim)

    transformer = LinearTransformer(
        in_channels=in_dim,
        hidden_channels=hidden_dim,
        out_channels= in_dim,
        num_layers=num_layers,
        num_heads=num_heads
    )

    output = transformer(dummy_input)

    print(f"input: {dummy_input.shape}")
    print(f"output: {output.shape}")  #  torch.Size([100, 128])
