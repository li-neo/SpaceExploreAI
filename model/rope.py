import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0, scaling_factor: float = 1.0):
    """
    预计算旋转位置编码的频率复数形式
    
    参数:
        dim: 头维度，必须是偶数
        max_seq_len: 最大序列长度
        theta: 旋转角度基值
        scaling_factor: 扩展序列长度的缩放因子

    返回:
        复数形式的频率张量，形状为 [max_seq_len, dim/2]
    """
    # 确保维度是偶数
    if dim % 2 != 0:
        raise ValueError(f"维度 ({dim}) 必须是偶数")
    
    # 如果使用了缩放因子，则对theta进行调整
    theta = theta * (1.0 / scaling_factor) ** (dim / (dim - 2))
    
    # 计算基本频率，只需要dim/2个频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    
    # 生成位置序列
    t = torch.arange(max_seq_len, dtype=torch.float)
    
    # 计算频率外积
    freqs = torch.outer(t, freqs)  # [max_seq_len, dim/2]
    
    # 使用欧拉公式将频率转换为复数形式: e^(i*θ) = cos(θ) + i*sin(θ)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, position_ids=None) -> torch.Tensor:
    """
    应用旋转位置编码到输入张量

        # q_rope: [batch_size,seq_len,num_heads,qk_rope_head_dim]
        # freqs_cis: [max_seq_len,qk_rope_head_dim]
        # position_ids: [batch_size,seq_len]
    
    参数:
        x: 输入张量 [batch_size, seq_len, n_heads, head_dim] 或 [batch_size, seq_len, head_dim]
        freqs_cis: 预计算的频率张量 [max_seq_len, dim/2]


        position_ids: 位置ID，如果None则使用默认顺序位置, 动态控制位置编码的索引映射
           position_ids的作用主要有两点：一是允许动态指定位置，适应非连续或特定顺序的位置；
           二是确保在不同批次或变长序列中正确应用旋转编码，维持模型的相对位置处理能力

    返回:
        应用了旋转位置编码的张量
    """
    # 保存原始数据类型
    dtype = x.dtype
    
    # 获取维度
    if len(x.shape) == 4:
        batch_size, seq_len, n_heads, head_dim = x.shape
    else:
        batch_size, seq_len, head_dim = x.shape
        n_heads = 1
    
    # 确保head_dim是偶数
    if head_dim % 2 != 0:
        raise ValueError(f"头维度 ({head_dim}) 必须是偶数")
    
    # 确保freqs_cis的维度与head_dim/2匹配
    if freqs_cis.shape[-1] != head_dim // 2:
        raise ValueError(f"频率张量的最后一个维度 ({freqs_cis.shape[-1]}) 必须等于头维度的一半 ({head_dim // 2})")
    
    # 如果提供了位置ID，则按照位置ID获取对应的频率
    if position_ids is not None:
        # 确保position_ids的形状正确
        if position_ids.shape[0] != batch_size or position_ids.shape[1] != seq_len:
            raise ValueError(f"位置ID的形状 {position_ids.shape} 必须匹配批量大小和序列长度 ({batch_size}, {seq_len})")
        
        # 获取对应位置的频率
        freqs_cis = freqs_cis[position_ids]  # [batch_size, seq_len, head_dim/2]
    else:
        # 否则取序列长度的频率
        if seq_len > freqs_cis.shape[0]:
            raise ValueError(f"序列长度 ({seq_len}) 超过了预计算的频率张量长度 ({freqs_cis.shape[0]})")
        freqs_cis = freqs_cis[:seq_len]  # [seq_len, head_dim/2]
        # 扩展到批量维度
        # freqs_cis: [seq_len, head_dim/2] -> [batch_size, seq_len, head_dim/2]
        freqs_cis = freqs_cis.unsqueeze(0).expand(batch_size, seq_len, head_dim // 2)  # [batch_size, seq_len, head_dim/2]
    
    # 将输入张量视为复数
    # x: [batch_size, seq_len, n_heads, head_dim] -> [batch_size, seq_len, n_heads, head_dim/2] 
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    
    # 调整频率张量的形状以便于广播
    if len(x.shape) == 4:
        # 对于多头情况，需要在头维度上扩展
        # 从 [batch_size, seq_len, head_dim/2] 到 [batch_size, seq_len, n_heads, head_dim/2]
        freqs_cis = freqs_cis.unsqueeze(2).expand(batch_size, seq_len, n_heads, head_dim // 2)
    
    # 复数乘法实现旋转
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    
    # 恢复原始数据类型
    # x_rotated: [batch_size, seq_len, n_heads, head_dim]
    return x_rotated.type(dtype)


class RotaryEmbedding(nn.Module):
    """
    旋转位置编码模块
    """
    def __init__(self, 
                 dim: int, 
                 max_seq_len: int = 128, 
                 theta: float = 10000.0,
                 scaling_factor: float = 1.0,
                 scaling_type: str = "linear"):
        """
        初始化旋转位置编码模块
        
        参数:
            dim: 头维度，必须是偶数
            max_seq_len: 最大序列长度
            theta: 旋转角度基值
            scaling_factor: 扩展序列长度的缩放因子
            scaling_type: 缩放类型，'linear'或'ntk'
        """
        super().__init__()
        
        # 确保维度是偶数
        if dim % 2 != 0:
            raise ValueError(f"维度 ({dim}) 必须是偶数")
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.scaling_factor = scaling_factor
        self.scaling_type = scaling_type
        
        # 预计算频率
        self.freqs_cis = precompute_freqs_cis(self.dim, self.max_seq_len, self.theta, self.scaling_factor)
        self.register_buffer("freqs_buffer", self.freqs_cis)
    
    def forward(self, x: torch.Tensor, position_ids=None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量 [batch_size, seq_len, n_heads, head_dim] 或 [batch_size, seq_len, head_dim]
            position_ids: 位置ID，形状为 [batch_size, seq_len]

        返回:
            应用了旋转位置编码的张量
        """
        return apply_rotary_emb(x, self.freqs_buffer, position_ids)


class NTKScaledRotaryEmbedding(RotaryEmbedding):
    """
    使用NTK缩放的旋转位置编码
    
    NTK (Neural Tangent Kernel) 缩放可以让模型处理比训练时更长的序列
    """
    def __init__(self, 
                 dim: int, 
                 max_seq_len: int = 4096, 
                 theta: float = 10000.0,
                 scaling_factor: float = 1.0,
                 original_max_len: int = 4096,
                 beta_fast: int = 32,
                 beta_slow: int = 1):
        """
        初始化NTK缩放的旋转位置编码
        
        参数:
            dim: 隐藏维度
            max_seq_len: 最大序列长度
            theta: 旋转角度基值
            scaling_factor: 扩展序列长度的缩放因子
            original_max_len: 原始最大长度
            beta_fast: 快速缩放参数
            beta_slow: 慢速缩放参数
        """
        self.original_max_len = original_max_len
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        
        super().__init__(dim=dim, 
                         max_seq_len=max_seq_len, 
                         theta=theta,
                         scaling_factor=scaling_factor)
    
    def _find_correction_dim(self, num_rotations, dim, base, max_seq_len):
        """
        计算旋转位置编码的校正维度
        
        参数:
            num_rotations: 旋转数
            dim: 维度
            base: 基值
            max_seq_len: 最大序列长度
            
        返回:
            校正维度
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))
    
    def _find_correction_range(self, low_rot, high_rot, dim, base, max_seq_len):
        """
        计算校正范围
        
        参数:
            low_rot: 最小旋转数
            high_rot: 最大旋转数
            dim: 维度
            base: 基值
            max_seq_len: 最大序列长度
            
        返回:
            校正范围的下限和上限
        """
        low = math.floor(self._find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(self._find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)
    
    def _linear_ramp_factor(self, min_val, max_val, dim):
        """
        计算线性渐变因子
        
        参数:
            min_val: 最小值
            max_val: 最大值
            dim: 维度
            
        返回:
            线性渐变张量
        """
        if min_val == max_val:
            max_val += 0.001
        
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
        return torch.clamp(linear_func, 0, 1)
    
    def precompute_freqs_cis(self):
        """
        使用NTK缩放预计算频率
        
        返回:
            频率张量
        """
        # 计算基本频率
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        
        # 如果序列长度超过原始长度，应用NTK缩放
        if self.max_seq_len > self.original_max_len:
            # 计算校正范围
            low, high = self._find_correction_range(
                self.beta_fast, 
                self.beta_slow, 
                self.dim, 
                self.theta, 
                self.original_max_len
            )
            
            # 计算平滑系数
            smooth = 1 - self._linear_ramp_factor(low, high, self.dim // 2)
            
            # 应用NTK缩放
            freqs = freqs / self.scaling_factor * (1 - smooth) + freqs * smooth
        
        # 生成位置序列
        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        
        # 计算频率外积
        freqs = torch.outer(t, freqs)
        
        # 转换为复数形式
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        
        return freqs_cis


# 测试代码
if __name__ == "__main__":
    # 测试参数
    batch_size = 2
    seq_len = 16
    n_heads = 4
    head_dim = 64
    
    # 创建测试输入
    x = torch.randn(batch_size, seq_len, n_heads, head_dim)
    
    # 测试标准旋转编码
    print("测试标准旋转编码...")
    rotary = RotaryEmbedding(dim=head_dim, max_seq_len=128)
    rotated_x = rotary(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {rotated_x.shape}")
    
    # 测试NTK缩放的旋转编码
    print("\n测试NTK缩放的旋转编码...")
    ntk_rotary = NTKScaledRotaryEmbedding(
        dim=head_dim, 
        max_seq_len=1024, 
        original_max_len=512,
        scaling_factor=2.0
    )
    ntk_rotated_x = ntk_rotary(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {ntk_rotated_x.shape}")
    
    # 检查是否保持输入尺寸不变
    assert x.shape == rotated_x.shape
    assert x.shape == ntk_rotated_x.shape
    print("\n所有测试通过！") 