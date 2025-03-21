import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List


class MLP(nn.Module):
    """
    多层感知机（MLP）模块，用作前馈网络
    """
    def __init__(self, 
                 hidden_size: int, 
                 intermediate_size: int,
                 activation_fn: str = "silu"):
        """
        初始化MLP
        
        参数:
            hidden_size: 隐藏层维度
            intermediate_size: 中间层维度
            activation_fn: 激活函数，可选 "relu", "gelu", "silu"
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # 激活函数
        if activation_fn == "relu":
            self.activation_fn = F.relu
        elif activation_fn == "gelu":
            self.activation_fn = F.gelu
        else:  # silu (swish) by default
            self.activation_fn = F.silu
            
        # 投影层
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w3.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量 [batch_size, seq_len, hidden_size]
            
        返回:
            输出张量 [batch_size, seq_len, hidden_size]
        """
        # SwiGLU激活
        return self.w2(self.activation_fn(self.w1(x)) * self.w3(x))


class Expert(nn.Module):
    """
    专家模块，用于混合专家（MoE）
    """
    def __init__(self,
                hidden_size: int,
                intermediate_size: int,
                activation_fn: str = "silu"):
        """
        初始化专家模块
        
        参数:
            hidden_size: 隐藏层维度
            intermediate_size: 中间层维度
            activation_fn: 激活函数，可选 "relu", "gelu", "silu"
        """
        super().__init__()
        
        # 创建MLP作为专家
        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_fn=activation_fn
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量 [batch_size, seq_len, hidden_size]
            
        返回:
            输出张量 [batch_size, seq_len, hidden_size]
        """
        return self.mlp(x)


class RoutingNetwork(nn.Module):
    """
    路由网络，用于MoE专家路由
    """
    def __init__(self,
                hidden_size: int,
                num_experts: int,
                num_experts_per_token: int = 2,
                routing_scale: float = 1.0,
                score_func: str = "softmax"):
        """
        初始化路由网络
        
        参数:
            hidden_size: 隐藏层维度
            num_experts: 专家数量
            num_experts_per_token: 每个token使用的专家数量
            routing_scale: 路由权重缩放因子
            score_func: 分数函数，"softmax"或"sigmoid"
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.routing_scale = routing_scale
        self.score_func = score_func
        
        # 路由器权重
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
        # 初始化权重
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        参数:
            x: 输入张量 [batch_size, seq_len, hidden_size] 或 [batch_size * seq_len, hidden_size]
            
        返回:
            routing_weights: 路由权重 [batch_size, seq_len, num_experts_per_token] 或 [batch_size * seq_len, num_experts_per_token]
            top_k_indices: 选中的专家索引 [batch_size, seq_len, num_experts_per_token] 或 [batch_size * seq_len, num_experts_per_token]
        """
        # 保存输入形状用于调试
        input_shape = x.shape
        is_2d_input = len(input_shape) == 2
        
        # x: [batch_size, seq_len, hidden_size] 或 [batch_size * seq_len, hidden_size]
        # scores: [batch_size, seq_len, num_experts] 或 [batch_size * seq_len, num_experts]
        scores = self.router(x)  # [batch_size, seq_len, num_experts] 或 [batch_size * seq_len, num_experts]
        
        # 原始分数用于之后的权重计算 - 使用detach().clone()断开计算图
        # original_scores: [batch_size, seq_len, num_experts] 或 [batch_size * seq_len, num_experts]
        original_scores = scores.detach().clone()
        
        # 应用分数函数
        if self.score_func == "softmax":
            scores = F.softmax(scores, dim=-1)
        else:  # sigmoid
            scores = torch.sigmoid(scores)
        
        # 选择前k个专家
        top_k_scores, top_k_indices = torch.topk(
            scores, k=self.num_experts_per_token, dim=-1, sorted=False
        )
        
        # 获取所选专家的原始分数
        routing_weights = original_scores.gather(dim=-1, index=top_k_indices)
        
        # 对路由权重进行归一化
        if self.score_func == "sigmoid":
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # 应用缩放因子
        routing_weights = routing_weights * self.routing_scale
        
        # 打印形状信息帮助调试
        print(f"RoutingNetwork输入形状: {input_shape}")
        print(f"路由权重形状: {routing_weights.shape}")
        print(f"专家索引形状: {top_k_indices.shape}")
        
        return routing_weights, top_k_indices


class MixtureOfExperts(nn.Module):
    """
    混合专家（Mixture of Experts, MoE）模块
    """
    def __init__(self,
                hidden_size: int,
                moe_intermediate_size: int,
                num_experts: int = 8,
                num_experts_per_token: int = 2,
                num_shared_experts: int = 1,
                shared_intermediate_size: int = None,
                activation_fn: str = "silu",
                routing_scale: float = 1.0,
                score_func: str = "softmax"):
        """
        初始化MoE模块
        
        参数:
            hidden_size: 隐藏层维度
            moe_intermediate_size: MoE中间层维度
            num_experts: 专家数量
            num_experts_per_token: 每个token使用的专家数量
            num_shared_experts: 共享专家数量
            shared_intermediate_size: 共享专家的中间层维度
            activation_fn: 激活函数
            routing_scale: 路由权重缩放因子
            score_func: 分数函数
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.num_shared_experts = num_shared_experts
        
        # 创建路由网络
        self.router = RoutingNetwork(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            routing_scale=routing_scale,
            score_func=score_func
        )
        
        # 创建专家集合
        self.experts = nn.ModuleList([
            Expert(
                hidden_size=hidden_size,
                intermediate_size=moe_intermediate_size,
                activation_fn=activation_fn
            ) for _ in range(num_experts)
        ])
        
        # 创建共享专家（如果有）
        if num_shared_experts > 0:
            if shared_intermediate_size is None:
                shared_intermediate_size = moe_intermediate_size
                
            self.shared_expert = MLP(
                hidden_size=hidden_size,
                intermediate_size=shared_intermediate_size * num_shared_experts,
                activation_fn=activation_fn
            )
    
    def forward(self, x: torch.Tensor, expert_choices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量 [batch_size, seq_len, hidden_size]
            expert_choices: 预选的专家索引（可选）
            
        返回:
            输出张量 [batch_size, seq_len, hidden_size]
        """
        # 保存原始形状
        original_shape = x.shape
        print(f"MoE输入形状: {original_shape}")
        
        # 重塑为二维方便处理
        # x_2d: [batch_size * seq_len, hidden_size]
        batch_size, seq_len = original_shape[0], original_shape[1]
        x_2d = x.reshape(-1, self.hidden_size)
        print(f"重塑后x_2d形状: {x_2d.shape}")
        
        # 如果没有提供专家选择，使用路由器
        # routing_weights: [batch_size * seq_len, num_experts]
        # expert_indices: [batch_size * seq_len, num_experts_per_token]
        if expert_choices is None:
            routing_weights, expert_indices = self.router(x_2d)
        else:
            routing_weights, expert_indices = expert_choices
            
        print(f"路由权重形状: {routing_weights.shape}")
        print(f"专家索引形状: {expert_indices.shape}")
        
        # 创建输出张量
        moe_out = torch.zeros_like(x_2d)
        
        # 对每个专家进行计算
        for expert_idx in range(self.num_experts):
            # 找出选择这个专家的所有token
            # token_indices在第一个维度，batch*seq中的索引
            # position_in_expert_list是在专家列表中的位置，第二个维度
            token_indices, position_in_expert_list = torch.where(expert_indices == expert_idx)
            
            if len(token_indices) > 0:
                # 提取token输入
                expert_input = x_2d[token_indices].clone()  # 使用clone()避免共享内存
                # 计算专家输出
                expert_output = self.experts[expert_idx](expert_input)
                # 获取路由权重并克隆避免共享计算图
                expert_weights = routing_weights[token_indices, position_in_expert_list].clone().unsqueeze(-1)
                # 加权求和
                moe_out[token_indices] += expert_output * expert_weights
        
        # 添加共享专家的输出
        if hasattr(self, "shared_expert"):
            shared_out = self.shared_expert(x_2d.clone())  # 使用clone()避免共享内存
            moe_out = moe_out + shared_out
        
        # 重塑为原始形状
        moe_out = moe_out.view(original_shape)
        print(f"MoE输出形状: {moe_out.shape}")
        
        return moe_out


# 测试代码
if __name__ == "__main__":
    # 测试参数
    batch_size = 2
    seq_len = 16
    hidden_size  = 256
    intermediate_size = 1024
    moe_intermediate_size = 1024
    num_experts = 8
    num_experts_per_token = 2
    
    # 创建测试输入
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # 测试MLP
    print("\n========== 测试MLP ==========")
    mlp = MLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
    mlp_out = mlp(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {mlp_out.shape}")
    
    # 测试专家
    print("\n========== 测试专家 ==========")
    expert = Expert(hidden_size=hidden_size, intermediate_size=intermediate_size)
    expert_out = expert(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {expert_out.shape}")
    
    # 测试路由网络
    print("\n========== 测试路由网络 ==========")
    router = RoutingNetwork(
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token
    )
    # 首先测试2D输入
    x_2d = x.reshape(-1, hidden_size)
    print(f"2D输入形状: {x_2d.shape}")
    
    routing_weights_2d, expert_indices_2d = router(x_2d)
    print(f"2D输入时路由权重形状: {routing_weights_2d.shape}")
    print(f"2D输入时专家索引形状: {expert_indices_2d.shape}")
    
    # 然后测试3D输入
    print(f"3D输入形状: {x.shape}")
    routing_weights, expert_indices = router(x)
    print(f"3D输入时路由权重形状: {routing_weights.shape}")
    print(f"3D输入时专家索引形状: {expert_indices.shape}")
    
    # 测试混合专家
    print("\n========== 测试混合专家 ==========")
    moe = MixtureOfExperts(
        hidden_size=hidden_size,
        moe_intermediate_size=moe_intermediate_size,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        num_shared_experts=1
    )
    moe_out = moe(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {moe_out.shape}")
    
    # 检查形状是否一致
    print("\n========== 检查形状一致性 ==========")
    shape_checks = []
    
    shape_checks.append(mlp_out.shape == x.shape)
    print(f"MLP输出形状与输入一致: {shape_checks[-1]}")
    
    shape_checks.append(expert_out.shape == x.shape)
    print(f"专家输出形状与输入一致: {shape_checks[-1]}")
    
    shape_checks.append(moe_out.shape == x.shape)
    print(f"MoE输出形状与输入一致: {shape_checks[-1]}")
    
    shape_checks.append(routing_weights.shape[-1] == num_experts_per_token)
    print(f"路由权重最后一维等于每个token的专家数: {shape_checks[-1]}")
    
    shape_checks.append(expert_indices.shape[-1] == num_experts_per_token)
    print(f"专家索引最后一维等于每个token的专家数: {shape_checks[-1]}")
    
    if all(shape_checks):
        print("\n所有形状检查通过！")
    else:
        print("\n形状检查失败！")
        for i, check in enumerate(shape_checks):
            if not check:
                print(f"  检查 {i+1} 失败")
    
    print("\n所有测试完成！") 