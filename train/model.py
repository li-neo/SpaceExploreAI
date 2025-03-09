import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        # 分头处理，并行计算
        Q = self.W_q(query).view(-1, self.num_heads, self.d_k)
        K = self.W_k(key).view(-1, self.num_heads, self.d_k)
        V = self.W_v(value).view(-1, self.num_heads, self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V).contiguous().view(query.size())
        return self.W_o(output)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 残差连接 + 层归一化
        x = self.norm1(x + self.attn(x, x, x, mask))
        x = self.norm2(x + self.ffn(x))
        return self.dropout(x)



class SparseAttention(MultiHeadAttention):
    def __init__(self, d_model, num_heads, window_size=3):
        super().__init__(d_model, num_heads)
        self.window_size = window_size

    def create_sparse_mask(self, seq_length):
        mask = torch.zeros((seq_length, seq_length))
        for i in range(seq_length):
            start = max(0, i - self.window_size)
            end = min(seq_length, i + self.window_size + 1)
            mask[i, start:end] = 1
        return mask.bool()

class ModelArgs:
    vocab_size: int = 10240
    model_dim: int = 128
    hidden_dim: int = 128
    num_heads: int = 8
    layers: int = 16
    feedforward_dim: int = 2048
    dropout: float = 0.2


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed = nn.Embedding(args.vocab_size, args.model_dim)
        self.pos_encoder = PositionalEncoding(args.model_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(args.model_dim, args.num_heads, args.feedforward_dim, dropout=args.dropout)
            for _ in range(args.layers)
        ])
        self.liner = nn.Linear(args.model_dim, args.vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.pos_encoder(self.embed(src))
        tgt = self.pos_encoder(self.embed(tgt))
        for block in self.blocks:
            src = block(src, src_mask)
            tgt = block(tgt, tgt_mask)
        return self.liner(tgt)




# class Transformer(nn.Module):
#     def __init__(self,args: ModelArgs):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         self.weight = nn.Parameter(torch.empty(args.embedding_dim,args.vocab_size))
#
#     def forward(self, tokens: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
#         for layer in self.layers:
#             tokens = layer(tokens, start_pos)
#
#         tokens = F.relu(tokens)
#         tokens = F.linear(tokens, self.weight)
#         return tokens
