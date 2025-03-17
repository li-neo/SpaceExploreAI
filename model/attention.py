import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rope import apply_rotary_emb
from log.logger import get_logger

logger = get_logger(__name__, log_file="attention.log")


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
    混合潜在注意力模块，支持三种注意力机制：
    1. 标准多头注意力 (Standard Multi-head Attention)
    2. 分组查询注意力 (Grouped-Query Attention, GQA)
    3. 吸收式注意力 (Absorb Attention)
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
                 dtype: str = "float16",
                 attention_type: str = "absorb"):  # 默认使用吸收式注意力
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
            dtype: 数据类型，可选 "float16"、"float32" 或 "bfloat16"
            attention_type: 注意力类型，可选 "standard"、"gqa" 或 "absorb"
        """
        super().__init__()
        
        # 验证参数
        if attention_type not in ["standard", "gqa", "absorb"]:
            raise ValueError(f"不支持的注意力类型: {attention_type}，必须是 'standard'、'gqa' 或 'absorb'")
        
        if dtype not in ["float16", "float32", "bfloat16"]:
            raise ValueError(f"不支持的数据类型: {dtype}，必须是 'float16'、'float32' 或 'bfloat16'")
        
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
        self.dtype_str = dtype
        
        # 设置注意力类型
        self.attention_type = attention_type
        self.use_absorb_attn = (attention_type == "absorb")
        self.use_gqa = (attention_type == "gqa")
        
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
        
        # 为GQA设置KV头数量
        if self.use_gqa:
            self.num_kv_heads = max(1, num_heads // 4)  # 确保至少有1个KV头
            self.num_kv_groups = num_heads // self.num_kv_heads
            if num_heads % self.num_kv_heads != 0:
                raise ValueError(f"注意力头数量 ({num_heads}) 必须能被KV头数量 ({self.num_kv_heads}) 整除")
        
        # 初始化缓存 - 使用空张量并指定dtype以确保类型一致性
        torch_dtype = self._get_torch_dtype(dtype)
        
        # 使用 torch.zeros 而非 torch.empty 以避免未初始化内存可能导致的不确定性
        if self.use_absorb_attn:
            # 对于吸收式注意力，缓存KV潜在投影和位置编码
            self.register_buffer("kv_cache", torch.zeros(max_batch_size, max_seq_len, kv_lora_rank, dtype=torch_dtype))
            self.register_buffer("pe_cache", torch.zeros(max_batch_size, max_seq_len, qk_rope_head_dim, dtype=torch_dtype))
        elif self.use_gqa:
            # 对于GQA注意力，缓存K和V，但使用较少的头
            self.register_buffer("k_cache", torch.zeros(max_batch_size, max_seq_len, self.num_kv_heads, self.qk_head_dim, dtype=torch_dtype))
            self.register_buffer("v_cache", torch.zeros(max_batch_size, max_seq_len, self.num_kv_heads, self.v_head_dim, dtype=torch_dtype))
        else:
            # 对于标准注意力，直接缓存K和V
            self.register_buffer("k_cache", torch.zeros(max_batch_size, max_seq_len, num_heads, self.qk_head_dim, dtype=torch_dtype))
            self.register_buffer("v_cache", torch.zeros(max_batch_size, max_seq_len, num_heads, self.v_head_dim, dtype=torch_dtype))
    
    def _get_torch_dtype(self, dtype_str):
        """将字符串数据类型转换为torch数据类型"""
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float16
        }
        return dtype_map.get(dtype_str, torch.float32)
    
    def _prepare_attention_mask(self, attention_mask, batch_size, seq_len, target_dim=4):
        """
        准备注意力掩码，确保维度正确
        
        参数:
            attention_mask: 输入的注意力掩码
            batch_size: 批量大小
            seq_len: 序列长度
            target_dim: 目标维度 (4 表示 [batch_size, 1, 1, seq_len])
            
        返回:
            处理后的注意力掩码
        """
        if attention_mask is None:
            return None
            
        # 如果是2D掩码 [batch_size, seq_len]，转换为4D
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            
        # 确保掩码覆盖整个序列长度
        if attention_mask.size(-1) < seq_len:
            padding = torch.zeros(
                (batch_size, *attention_mask.shape[1:-1], seq_len - attention_mask.size(-1)),
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            attention_mask = torch.cat([attention_mask, padding], dim=-1)
            
        # 确保维度正确
        while attention_mask.dim() < target_dim:
            attention_mask = attention_mask.unsqueeze(1)
            
        return attention_mask
    
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
            attention_mask: 注意力掩码，形状为 [batch_size, seq_len] 或 [batch_size, 1, seq_len, seq_len]
            position_ids: 位置ID，形状为 [batch_size, seq_len]
            start_pos: 起始位置（用于增量解码）
            output_attentions: 是否输出注意力权重

        返回:
            注意力输出，形状为 [batch_size, seq_len, hidden_size]
            如果 output_attentions=True，则还返回注意力权重
        """
        batch_size, seq_len, _ = hidden_states.size()
        end_pos = start_pos + seq_len
        
        # 检查缓存大小是否足够
        if end_pos > self.max_seq_len:
            raise ValueError(f"序列长度 ({end_pos}) 超过了最大缓存长度 ({self.max_seq_len})")
        
        if batch_size > self.max_batch_size:
            raise ValueError(f"批量大小 ({batch_size}) 超过了最大批量大小 ({self.max_batch_size})")
        
        # 预处理注意力掩码，确保格式一致
        attention_mask = self._prepare_attention_mask(attention_mask, batch_size, end_pos)
        
        # 如果使用标准注意力机制
        if self.attention_type == "standard":
            return self._forward_standard(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                position_ids=position_ids,
                start_pos=start_pos,
                output_attentions=output_attentions
            )
        
        # 如果使用分组查询注意力 (GQA)
        elif self.attention_type == "gqa":
            return self._forward_gqa(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                position_ids=position_ids,
                start_pos=start_pos,
                output_attentions=output_attentions
            )
        
        # 如果使用吸收式注意力
        elif self.attention_type == "absorb":
            return self._forward_absorb(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                position_ids=position_ids,
                start_pos=start_pos,
                output_attentions=output_attentions
            )
        
        else:
            raise ValueError(f"不支持的注意力类型: {self.attention_type}")
    
    def _forward_standard(self,
                         hidden_states: torch.Tensor,
                         freqs_cis: torch.Tensor,
                         attention_mask: torch.Tensor = None,
                         position_ids: torch.Tensor = None,
                         start_pos: int = 0,
                         output_attentions: bool = False):
        """标准多头注意力的前向传播"""
        batch_size, seq_len, _ = hidden_states.size()
        end_pos = start_pos + seq_len
        
        # 处理缓存
        past_key_value = None
        if start_pos > 0:
            past_key_value = (
                self.k_cache[:batch_size, :start_pos],
                self.v_cache[:batch_size, :start_pos]
            )
        
        # 调用标准注意力
        outputs = self.attention(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=True
        )
        
        # 更新缓存
        attn_output, current_key_value = outputs[:2]
        if current_key_value is not None:
            k, v = current_key_value
            self.k_cache[:batch_size, :end_pos] = k
            self.v_cache[:batch_size, :end_pos] = v
        
        # 处理返回值
        if output_attentions and len(outputs) > 2:
            return attn_output, outputs[2]
        return attn_output
    
    def _forward_gqa(self,
                    hidden_states: torch.Tensor,
                    freqs_cis: torch.Tensor,
                    attention_mask: torch.Tensor = None,
                    position_ids: torch.Tensor = None,
                    start_pos: int = 0,
                    output_attentions: bool = False):
        """
        分组查询注意力 (GQA) 的前向传播
        
        GQA使用较少的K和V头，每个K/V头被多个Q头共享，减少计算量和内存使用
        """
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
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, position_ids)
        
        # 计算KV的第二次投影
        kv = self.attention.kv_norm(kv)
        kv = self.attention.kv_proj_b(kv)
        
        # 重塑KV为多头形式
        kv = kv.view(batch_size, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        
        # 分离K和V
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        
        # 重塑为GQA格式（减少KV头数）
        # 将Q头分组，每组对应一个KV头
        k_nope = k_nope.view(batch_size, seq_len, self.num_kv_groups, self.num_kv_heads, self.qk_nope_head_dim)
        k_nope = k_nope.mean(dim=2)  # 合并同一KV头对应的多个Q头
        
        k_pe = k_pe.squeeze(2).view(batch_size, seq_len, 1, self.qk_rope_head_dim)
        k_pe = k_pe.expand(-1, -1, self.num_kv_heads, -1)  # 扩展到所有KV头
        
        v = v.view(batch_size, seq_len, self.num_kv_groups, self.num_kv_heads, self.v_head_dim)
        v = v.mean(dim=2)  # 合并同一KV头对应的多个Q头
        
        # 合并K的两部分
        k = torch.cat([k_nope, k_pe], dim=-1)
        
        # 更新缓存
        self.k_cache[:batch_size, start_pos:end_pos] = k
        self.v_cache[:batch_size, start_pos:end_pos] = v
        
        # 如果是增量解码，使用完整的缓存
        if start_pos > 0:
            k = self.k_cache[:batch_size, :end_pos]
            v = self.v_cache[:batch_size, :end_pos]
        
        # 重新合并Q的两部分
        q = torch.cat([q_nope, q_pe], dim=-1)
        
        # 将Q头分组以匹配KV头
        # 每个KV头对应多个Q头
        q_grouped = q.view(batch_size, seq_len, self.num_kv_groups, self.num_kv_heads, self.qk_head_dim)
        
        # 计算注意力分数 [batch_size, seq_len, num_kv_groups, num_kv_heads, seq_len]
        attention_scores = torch.einsum("bsgnd,bthd->bsgnth", q_grouped, k) * self.softmax_scale
        
        # 应用注意力掩码
        if attention_mask is not None:
            # 确保掩码维度正确 [batch_size, 1, 1, 1, seq_len]
            attention_mask = self._prepare_attention_mask(attention_mask, batch_size, end_pos, target_dim=5)
            attention_scores = attention_scores + attention_mask
        
        # 应用softmax获取注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1, dtype=torch.float32).type_as(q)
        
        # 应用dropout
        attention_weights = self.attention.attn_dropout(attention_weights)
        
        # 计算注意力输出 [batch_size, seq_len, num_kv_groups, num_kv_heads, v_head_dim]
        attn_output = torch.einsum("bsgnth,bthd->bsgnd", attention_weights, v)
        
        # 重塑输出
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.v_head_dim)
        
        # 应用输出投影
        attn_output = self.attention.out_proj(attn_output)
        
        # 处理返回值
        if output_attentions:
            return attn_output, attention_weights
        return attn_output
    
    def _forward_absorb(self,
                       hidden_states: torch.Tensor,
                       freqs_cis: torch.Tensor,
                       attention_mask: torch.Tensor = None,
                       position_ids: torch.Tensor = None,
                       start_pos: int = 0,
                       output_attentions: bool = False):
        """
        吸收式注意力的前向传播
        
        吸收式注意力通过缓存中间表示而非最终的K和V来优化计算
        """
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
        
        # 验证权重矩阵分割是否正确
        k_proj_size = self.qk_nope_head_dim
        v_proj_size = self.v_head_dim
        
        # 确保权重矩阵的第二维与预期一致
        assert wkv_b.shape[1] == k_proj_size + v_proj_size, \
            f"权重矩阵维度不匹配: {wkv_b.shape[1]} != {k_proj_size + v_proj_size}"
        
        # 计算Q和KV的乘积
        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :k_proj_size])
        
        # 计算注意力分数
        attn_scores = (
            torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:batch_size, :end_pos]) +
            torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:batch_size, :end_pos])
        ) * self.softmax_scale
        
        # 应用注意力掩码
        if attention_mask is not None:
            # 确保掩码维度正确 [batch_size, 1, 1, seq_len]
            if attention_mask.dim() == 4:
                # 转换为 [batch_size, 1, seq_len]
                attention_mask = attention_mask.squeeze(2)
            attn_scores = attn_scores + attention_mask
        
        # 应用softmax获取注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).type_as(q)
        
        # 应用dropout
        attn_weights = self.attention.attn_dropout(attn_weights)
        
        # 计算注意力输出
        x = torch.einsum("bsht,btc->bshc", attn_weights, self.kv_cache[:batch_size, :end_pos])
        x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -v_proj_size:])
        
        # 重塑输出并应用输出投影
        x = x.reshape(batch_size, seq_len, self.num_heads * self.v_head_dim)
        x = self.attention.out_proj(x)
        
        # 处理返回值
        if output_attentions:
            return x, attn_weights
        return x


# 测试代码
if __name__ == "__main__":
    import torch
    from model.rope import precompute_freqs_cis
    
    # 测试参数
    batch_size = 16
    seq_len = 128
    hidden_size = 256
    num_heads = 4
    qk_nope_head_dim = 32
    qk_rope_head_dim = 32
    v_head_dim = 64
    
    # 创建测试输入
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # 预计算频率 - 修复：确保频率张量的维度与序列长度匹配
    max_seq_len = 128  # 设置足够大的最大序列长度
    freqs_cis = precompute_freqs_cis(qk_rope_head_dim, max_seq_len)
    
    # 创建位置ID
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    
    # 测试多头潜在注意力
    logger.info(f"测试多头潜在注意力...")
    mla = MultiLatentAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim
    )
    output = mla(hidden_states, freqs_cis, position_ids=position_ids)[0]
    logger.info(f"输入形状: {hidden_states.shape}")
    logger.info(f"输出形状: {output.shape}")
    
    # 测试混合潜在注意力 - 修复：确保批量大小不超过最大批量大小
    logger.info(f"测试混合潜在注意力...")
    test_batch_size = 4  # 确保不超过max_batch_size
    test_hidden_states = hidden_states[:test_batch_size]
    test_position_ids = position_ids[:test_batch_size]
    
    # 测试标准注意力
    logger.info(f"测试标准注意力...")
    standard_mla = MixedLatentAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        max_batch_size=16,
        max_seq_len=256,
        dtype="float32",
        attention_type="standard"
    )
    standard_output = standard_mla(test_hidden_states, freqs_cis, position_ids=test_position_ids)
    logger.info(f"标准注意力输出形状: {standard_output.shape}")
    
    # 测试GQA注意力
    logger.info(f"测试GQA注意力...")
    gqa_mla = MixedLatentAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        max_batch_size=16,
        max_seq_len=256,
        dtype="float32",
        attention_type="gqa"
    )
    gqa_output = gqa_mla(test_hidden_states, freqs_cis, position_ids=test_position_ids)
    logger.info(f"GQA注意力输出形状: {gqa_output.shape}")
    
    # 测试吸收式注意力
    logger.info(f"测试吸收式注意力...")
    absorb_mla = MixedLatentAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        max_batch_size=16,
        max_seq_len=256,
        dtype="float32",
        attention_type="absorb"
    )
    absorb_output = absorb_mla(test_hidden_states, freqs_cis, position_ids=test_position_ids)
    logger.info(f"吸收式注意力输出形状: {absorb_output.shape}")
    
    # 检查输出尺寸是否正确
    assert output.shape == hidden_states.shape
    assert standard_output.shape == test_hidden_states.shape
    assert gqa_output.shape == test_hidden_states.shape
    assert absorb_output.shape == test_hidden_states.shape
    logger.info("所有测试通过！") 