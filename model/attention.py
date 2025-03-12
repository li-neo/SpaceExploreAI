import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rotary_emb


class MultiLatentAttention(nn.Module):
    """
    多头潜在注意力（Multi-head Latent Attention）模块
    
    该模块实现了一种高效的注意力机制，结合了旋转位置编码和潜在投影
    """
    
    def __init__(self, 
                 hidden_size: int, 
                 num_heads: int,
                 qk_nope_head_dim: int = 128,
                 qk_rope_head_dim: int = 64,
                 v_head_dim: int = 128,
                 dropout: float = 0.0,
                 q_lora_rank: int = 0,
                 kv_lora_rank: int = 512,
                 attention_scale_factor: float = 1.0):
        """
        初始化多头潜在注意力模块
        
        参数:
            hidden_size: 隐藏层维度
            num_heads: 注意力头数量
            qk_nope_head_dim: 不使用旋转位置编码的Q/K头维度
            qk_rope_head_dim: 使用旋转位置编码的Q/K头维度
            v_head_dim: 值向量的头维度
            dropout: Dropout比率
            q_lora_rank: Q的低秩适应维度，0表示不使用低秩适应
            kv_lora_rank: K和V的低秩适应维度
            attention_scale_factor: 注意力缩放因子
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.dropout = dropout
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        
        # 计算缩放因子
        self.softmax_scale = self.qk_head_dim ** -0.5 * attention_scale_factor
        
        # Q投影层，根据是否使用低秩适应选择不同的实现
        if self.q_lora_rank == 0:
            # 直接使用全尺寸投影
            self.q_proj = nn.Linear(hidden_size, num_heads * self.qk_head_dim, bias=False)
        else:
            # 使用低秩适应
            self.q_proj_a = nn.Linear(hidden_size, self.q_lora_rank, bias=False)
            self.q_norm = nn.LayerNorm(self.q_lora_rank)
            self.q_proj_b = nn.Linear(self.q_lora_rank, num_heads * self.qk_head_dim, bias=False)
        
        # KV低秩投影层
        self.kv_proj_a = nn.Linear(hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_norm = nn.LayerNorm(self.kv_lora_rank)
        # KV的第二个投影
        self.kv_proj_b = nn.Linear(self.kv_lora_rank, num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        
        # 输出投影层
        self.out_proj = nn.Linear(num_heads * self.v_head_dim, hidden_size, bias=False)
        
        # Dropout层
        self.attn_dropout = nn.Dropout(dropout)
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1 / math.sqrt(2))
        
        if self.q_lora_rank == 0:
            nn.init.xavier_uniform_(self.q_proj.weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_a.weight)
            nn.init.xavier_uniform_(self.q_proj_b.weight)
        
        nn.init.xavier_uniform_(self.kv_proj_a.weight)
        nn.init.xavier_uniform_(self.kv_proj_b.weight)
    
    def forward(self, 
               hidden_states: torch.Tensor, 
               freqs_cis: torch.Tensor,
               attention_mask: torch.Tensor = None,
               position_ids: torch.Tensor = None,
               past_key_value = None,
               output_attentions: bool = False,
               use_cache: bool = False) -> tuple:
        """
        前向传播
        
        参数:
            hidden_states: 输入张量，形状为 [batch_size, seq_len, hidden_size]
            freqs_cis: 旋转位置编码的频率张量
            attention_mask: 注意力掩码
            position_ids: 位置ID
            past_key_value: 用于缓存的过去KV状态
            output_attentions: 是否输出注意力权重
            use_cache: 是否使用KV缓存

        返回:
            (attn_output, attention_weights, past_key_value) 的元组
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # 计算Q投影
        if self.q_lora_rank == 0:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_proj_b(self.q_norm(self.q_proj_a(hidden_states)))
        
        # 重塑Q为多头形式 [batch_size, seq_len, num_heads, qk_head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.qk_head_dim)
        
        # 分离不使用位置编码和使用位置编码的部分
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # 应用旋转位置编码到需要的部分
        q_rope = apply_rotary_emb(q_rope, freqs_cis, position_ids)
        
        # 计算KV投影
        kv = self.kv_proj_a(hidden_states)
        kv, k_rope = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        
        # 应用旋转位置编码到k_rope
        k_rope = apply_rotary_emb(k_rope.unsqueeze(2), freqs_cis, position_ids)
        
        # 计算KV的第二次投影
        kv = self.kv_norm(kv)
        kv = self.kv_proj_b(kv)
        kv = kv.view(batch_size, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        
        # 分离K和V
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        
        # 合并K的两部分
        k = torch.cat([k_nope, k_rope.expand(-1, -1, self.num_heads, -1)], dim=-1)
        
        # 处理过去的键值对（用于解码器）
        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=1)
            v = torch.cat([past_value, v], dim=1)
        
        # 准备当前的键值对用于缓存
        if use_cache:
            current_key_value = (k, v)
        else:
            current_key_value = None
        
        # 重新合并Q的两部分
        q = torch.cat([q_nope, q_rope], dim=-1)
        
        # 计算注意力分数
        # [batch_size, num_heads, seq_len, seq_len]
        attention_scores = torch.einsum("bthd,bshd->btsh", q, k) * self.softmax_scale
        
        # 应用注意力掩码
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # 应用softmax获取注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1, dtype=torch.float32).type_as(q)
        
        # 应用dropout
        attention_weights = self.attn_dropout(attention_weights)
        
        # 计算注意力输出 [batch_size, seq_len, num_heads, v_head_dim]
        attn_output = torch.einsum("btsh,bshd->bthd", attention_weights, v)
        
        # 重塑输出并应用输出投影
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.v_head_dim)
        attn_output = self.out_proj(attn_output)
        
        # 处理返回值
        outputs = (attn_output, current_key_value)
        if output_attentions:
            outputs += (attention_weights,)
            
        return outputs


class MixedLatentAttention(nn.Module):
    """
    混合潜在注意力模块，在多头潜在注意力的基础上增加了更多优化
    """
    def __init__(self, 
                 hidden_size: int, 
                 num_heads: int,
                 qk_nope_head_dim: int = 128,
                 qk_rope_head_dim: int = 64,
                 v_head_dim: int = 128,
                 dropout: float = 0.0,
                 q_lora_rank: int = 0,
                 kv_lora_rank: int = 512,
                 attention_scale_factor: float = 1.0,
                 max_batch_size: int = 32,
                 max_seq_len: int = 4096,
                 use_absorb_attn: bool = True):
        """
        初始化混合潜在注意力模块
        
        参数:
            hidden_size: 隐藏层维度
            num_heads: 注意力头数量
            qk_nope_head_dim: 不使用旋转位置编码的Q/K头维度
            qk_rope_head_dim: 使用旋转位置编码的Q/K头维度
            v_head_dim: 值向量的头维度
            dropout: Dropout比率
            q_lora_rank: Q的低秩适应维度，0表示不使用低秩适应
            kv_lora_rank: K和V的低秩适应维度
            attention_scale_factor: 注意力缩放因子
            max_batch_size: 最大批量大小（用于缓存）
            max_seq_len: 最大序列长度（用于缓存）
            use_absorb_attn: 是否使用吸收式注意力机制
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.dropout = dropout
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.use_absorb_attn = use_absorb_attn
        
        # 计算缩放因子
        self.softmax_scale = self.qk_head_dim ** -0.5 * attention_scale_factor
        
        # 多头潜在注意力层
        self.attention = MultiLatentAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            dropout=dropout,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            attention_scale_factor=attention_scale_factor
        )
        
        # 初始化缓存
        if self.use_absorb_attn:
            # 对于吸收式注意力，缓存KV潜在投影和位置编码
            self.register_buffer("kv_cache", torch.zeros(max_batch_size, max_seq_len, kv_lora_rank))
            self.register_buffer("pe_cache", torch.zeros(max_batch_size, max_seq_len, qk_rope_head_dim))
        else:
            # 对于标准注意力，直接缓存K和V
            self.register_buffer("k_cache", torch.zeros(max_batch_size, max_seq_len, num_heads, self.qk_head_dim))
            self.register_buffer("v_cache", torch.zeros(max_batch_size, max_seq_len, num_heads, self.v_head_dim))
    
    def forward(self, 
               hidden_states: torch.Tensor, 
               freqs_cis: torch.Tensor,
               attention_mask: torch.Tensor = None,
               position_ids: torch.Tensor = None,
               start_pos: int = 0,
               output_attentions: bool = False):
        """
        前向传播
        
        参数:
            hidden_states: 输入张量，形状为 [batch_size, seq_len, hidden_size]
            freqs_cis: 旋转位置编码的频率张量
            attention_mask: 注意力掩码
            position_ids: 位置ID
            start_pos: 起始位置（用于增量解码）
            output_attentions: 是否输出注意力权重

        返回:
            注意力输出
        """
        # 如果使用标准注意力机制
        if not self.use_absorb_attn:
            return self.attention(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions
            )[0]
        
        # 以下是吸收式注意力的实现
        batch_size, seq_len, _ = hidden_states.size()
        end_pos = start_pos + seq_len
        
        # 计算Q投影
        if self.q_lora_rank == 0:
            q = self.attention.q_proj(hidden_states)
        else:
            q = self.attention.q_proj_b(
                self.attention.q_norm(
                    self.attention.q_proj_a(hidden_states)
                )
            )
        
        # 重塑Q为多头形式 [batch_size, seq_len, num_heads, qk_head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.qk_head_dim)
        
        # 分离不使用位置编码和使用位置编码的部分
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # 应用旋转位置编码到需要的部分
        q_pe = apply_rotary_emb(q_pe, freqs_cis, position_ids)
        
        # 计算KV投影
        kv = self.attention.kv_proj_a(hidden_states)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        
        # 应用旋转位置编码到k_pe
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, position_ids).squeeze(2)
        
        # 更新KV缓存
        self.kv_cache[:batch_size, start_pos:end_pos] = self.attention.kv_norm(kv)
        self.pe_cache[:batch_size, start_pos:end_pos] = k_pe
        
        # 获取KV投影的权重
        wkv_b = self.attention.kv_proj_b.weight.view(self.num_heads, -1, self.kv_lora_rank)
        
        # 计算Q和KV的乘积
        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
        
        # 计算注意力分数
        attn_scores = (
            torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:batch_size, :end_pos]) +
            torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:batch_size, :end_pos])
        ) * self.softmax_scale
        
        # 应用注意力掩码
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask.unsqueeze(1)
        
        # 应用softmax获取注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).type_as(q)
        
        # 应用dropout
        attn_weights = self.attention.attn_dropout(attn_weights)
        
        # 计算注意力输出
        x = torch.einsum("bsht,btc->bshc", attn_weights, self.kv_cache[:batch_size, :end_pos])
        x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        
        # 重塑输出并应用输出投影
        x = x.reshape(batch_size, seq_len, self.num_heads * self.v_head_dim)
        x = self.attention.out_proj(x)
        
        return x


# 测试代码
if __name__ == "__main__":
    import torch
    from .rope import precompute_freqs_cis
    
    # 测试参数
    batch_size = 2
    seq_len = 16
    hidden_size = 768
    num_heads = 12
    qk_nope_head_dim = 32
    qk_rope_head_dim = 32
    v_head_dim = 64
    
    # 创建测试输入
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # 预计算频率
    freqs_cis = precompute_freqs_cis(qk_rope_head_dim * 2, 128)
    
    # 测试多头潜在注意力
    print("测试多头潜在注意力...")
    mla = MultiLatentAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim
    )
    output = mla(hidden_states, freqs_cis)[0]
    print(f"输入形状: {hidden_states.shape}")
    print(f"输出形状: {output.shape}")
    
    # 测试混合潜在注意力
    print("\n测试混合潜在注意力...")
    mixed_mla = MixedLatentAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        max_batch_size=4,
        max_seq_len=128
    )
    mixed_output = mixed_mla(hidden_states, freqs_cis)
    print(f"输入形状: {hidden_states.shape}")
    print(f"输出形状: {mixed_output.shape}")
    
    # 检查输出尺寸是否正确
    assert output.shape == hidden_states.shape
    assert mixed_output.shape == hidden_states.shape
    print("\n所有测试通过！") 