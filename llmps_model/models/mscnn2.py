"""
优化后的多尺度卷积神经网络，严格对齐论文《LLM-PS》设计
主要改进：
1. 修复小波变换维度问题
2. 优化通道分配策略
3. 增强边界补偿机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from log.logger import get_logger

logger = get_logger("mscnn_v2", "llmps_model.log")

class Conv1dBlock(nn.Module):
    """修正空洞卷积的padding计算"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 dilation=1, use_bn=True):
        super().__init__()
        # 自动计算所需padding以保持尺寸不变
        padding = (kernel_size + (kernel_size-1)*(dilation-1)) // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else nn.Identity()
        self.activation = nn.GELU()
        
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class MSBlock(nn.Module):
    """确保所有分支输出相同维度"""
    def __init__(self, channels, num_branches=4, seq_len=96):
        super().__init__()
        # 动态通道分配（确保总和等于输入通道数）
        self.branch_channels = [channels // num_branches] * num_branches
        self.branch_channels[-1] += channels % num_branches  # 处理余数
        
        # 可学习残差缩放因子
        self.alpha = nn.Parameter(torch.ones(1))
        
        # 多分支初始化（保持输出维度一致）
        self.branches = nn.ModuleList([
            nn.Sequential(
                Conv1dBlock(
                    c, c, 
                    kernel_size=3, 
                    dilation=2**i  # 指数增长的空洞率
                ),
                TemporalPatternDecoupling(channels=c)
            )
            for i, c in enumerate(self.branch_channels)
        ])
        
        # 维度校验层
        self.dim_check = nn.ModuleList([
            nn.Identity() if i == 0 else 
            nn.AdaptiveAvgPool1d(seq_len) for i in range(num_branches)
        ])
        
        # 特征融合模块
        self.fusion = nn.Sequential(
            nn.Conv1d(2*channels, channels, 1),
            nn.ReLU()
        )
        
    def forward(self, x):
        identity = x
        
        # 通道划分（确保精确分割）
        split_x = torch.split(x, self.branch_channels, dim=1)
        
        # 多分支处理
        short_terms, long_terms = [], []
        for branch, x_part, check in zip(self.branches, split_x, self.dim_check):
            s, l = branch(x_part)
            # 统一维度
            s = check(s)
            l = check(l)
            short_terms.append(s)
            long_terms.append(l)
        
        # 模式组装前统一维度
        target_dim = short_terms[0].shape[-1]
        short_terms = [F.interpolate(s, size=target_dim, mode='linear') for s in short_terms]
        long_terms = [F.interpolate(l, size=target_dim, mode='linear') for l in long_terms]
        
        # 模式组装（级联策略）
        for b in range(1, len(self.branches)):
            short_terms[b] += short_terms[b-1]
        for b in reversed(range(len(self.branches)-1)):
            long_terms[b] += long_terms[b+1]
            
        # 特征融合（维度校验）
        fused = self.fusion(torch.cat([
            torch.cat(short_terms, dim=1),
            torch.cat(long_terms, dim=1)
        ], dim=1))
        
        return fused + self.alpha * identity

class MSCNN(nn.Module):
    """优化后的多尺度主干网络"""
    def __init__(self, in_channels=64, base_channels=128, ms_blocks=3, 
                 num_branches=4, seq_len=96, output_dim=512):
        super().__init__()
        # 初始嵌入层（扩大感受野）
        self.embed = Conv1dBlock(
            in_channels, base_channels, 
            kernel_size=3
        )
        
        # 多尺度块堆叠
        self.blocks = nn.ModuleList()
        current_channels = base_channels
        for i in range(ms_blocks):
            self.blocks.append(
                MSBlock(current_channels, num_branches, seq_len//(2**i))
            )
            if i < ms_blocks-1:
                self.blocks.append(nn.MaxPool1d(2))
                current_channels *= 2
        
        # 时空注意力（增强特征选择）
        self.attention = nn.Sequential(
            nn.Conv1d(current_channels, current_channels*2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(current_channels, 1, 1)
        )
        
        # 输出投影
        self.projection = nn.Linear(current_channels, output_dim)
        
    def forward(self, x):
        x = self.embed(x)
        
        # 多尺度特征提取
        for block in self.blocks:
            x = block(x)
        
        # 注意力加权（时空特征选择）
        attn_weights = torch.softmax(self.attention(x), dim=-1)
        x = (x * attn_weights).sum(dim=-1)
        
        return self.projection(x)

class MSCNNWithSemantic(nn.Module):
    """支持语义融合的增强版本"""
    def __init__(self, output_dim=512, num_heads=8):
        super().__init__()
        self.cnn = MSCNN(output_dim=output_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x, semantic_embed):
        cnn_feat = self.cnn(x)
        
        # 交叉注意力（时序特征与语义对齐）
        attn_out, _ = self.cross_attn(
            query=cnn_feat.unsqueeze(1),
            key=semantic_embed.unsqueeze(1),
            value=semantic_embed.unsqueeze(1)
        )
        
        return self.norm(attn_out.squeeze(1) + cnn_feat)

if __name__ == "__main__":
    # 验证改进模型
    model = MSCNN(seq_len=96)
    test_input = torch.randn(32, 64, 96)
    output = model(test_input)
    logger.info(f"输出维度: {output.shape}")  # [32, 512]
    
    # 测试语义融合
    semantic_model = MSCNNWithSemantic()
    semantic_embed = torch.randn(32, 512)
    fused_output = semantic_model(test_input, semantic_embed)
    logger.info(f"融合特征维度: {fused_output.shape}")  # [32, 512]