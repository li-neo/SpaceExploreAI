"""
多尺度卷积神经网络 (Multi-Scale Convolutional Neural Network)
用于从时间序列数据中提取多尺度特征，捕获短期波动和长期趋势

基于论文《LLM-PS: Empowering Large Language Models for Time Series Forecasting with Temporal Patterns and Semantics》实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
import os
import sys
from log.logger import get_logger

# 创建logger
logger = get_logger("mscnn", "llmps_model.log")

class Conv1dBlock(nn.Module):
    """
    基础一维卷积块
    
    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核大小
        stride (int): 步长
        padding (int): 填充大小
        use_bn (bool): 是否使用批归一化
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=True):
        super(Conv1dBlock, self).__init__()
        
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
        
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm1d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        """前向传播"""
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.relu(x)
        return x


class MSBlock(nn.Module):
    """
    多尺度块 (Multi-Scale Block)
    包含不同尺度的一维卷积操作，用于捕获不同尺度的时间特征
    
    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        scales (list): 不同尺度的卷积核大小列表
    """
    def __init__(self, in_channels, out_channels, scales=[3, 5, 7, 9]):
        super(MSBlock, self).__init__()
        
        # 为每个尺度创建一个卷积分支
        self.branches = nn.ModuleList()
        for scale in scales:
            padding = scale // 2  # 保持输出长度不变
            self.branches.append(
                Conv1dBlock(in_channels, out_channels, kernel_size=scale, padding=padding)
            )
        
        # 融合不同尺度的特征
        self.fusion = Conv1dBlock(out_channels * len(scales), out_channels, kernel_size=1, padding=0)
        
    def forward(self, x):
        """前向传播"""
        # 获取每个分支的输出
        branch_outputs = [branch(x) for branch in self.branches]
        
        # 沿通道维度拼接
        concat_features = torch.cat(branch_outputs, dim=1)
        
        # 融合特征
        output = self.fusion(concat_features)
        
        return output


class TemporalPatternDecoupling(nn.Module):
    """
    时间模式解耦模块
    使用小波变换将时间序列解耦为高频（短期波动）和低频（长期趋势）组件
    
    参数:
        wavelet (str): 小波类型
        level (int): 分解级别
    """
    def __init__(self, wavelet='db4', level=1):
        super(TemporalPatternDecoupling, self).__init__()
        self.wavelet = wavelet
        self.level = level
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (tensor): 输入时间序列 [batch_size, channels, seq_len]
            
        返回:
            tuple: (低频分量, 高频分量)
        """
        batch_size, channels, seq_len = x.shape
        device = x.device
        
        # 将张量转为CPU上的NumPy数组进行小波变换
        x_np = x.detach().cpu().numpy()
        
        # 初始化结果
        low_freq = np.zeros_like(x_np)
        high_freq = np.zeros_like(x_np)
        
        # 对每个样本和通道分别进行小波变换
        for b in range(batch_size):
            for c in range(channels):
                # 执行小波分解
                coeffs = pywt.wavedec(x_np[b, c], self.wavelet, level=self.level)
                
                # 分离低频和高频分量
                approx_coeffs = coeffs[0]
                detail_coeffs = coeffs[1:]
                
                # 重构低频分量 (使用逆小波变换)
                # 保留近似系数，将细节系数设为0
                low_coeffs = [approx_coeffs] + [None] * len(detail_coeffs)
                low_freq[b, c] = pywt.waverec(low_coeffs, self.wavelet)[:seq_len]
                
                # 重构高频分量 (使用逆小波变换)
                # 将近似系数设为0，保留细节系数
                high_coeffs = [np.zeros_like(approx_coeffs)]
                for i, detail in enumerate(detail_coeffs):
                    high_coeffs.append(detail)
                high_freq[b, c] = pywt.waverec(high_coeffs, self.wavelet)[:seq_len]
                
                # 确保分解是完整的: 原始 = 低频 + 高频 (验证)
                residual = x_np[b, c] - (low_freq[b, c] + high_freq[b, c])
                if np.abs(residual).max() > 1e-10:
                    # 如果残差较大，微调高频分量以确保精确分解
                    high_freq[b, c] += residual
        
        # 转回PyTorch张量并移到原始设备
        low_freq_tensor = torch.from_numpy(low_freq).to(device)
        high_freq_tensor = torch.from_numpy(high_freq).to(device)
        
        return low_freq_tensor, high_freq_tensor


class TemporalPatternAssembling(nn.Module):
    """
    时间模式组装模块
    通过全局-局部和局部-全局交互增强时间特征
    
    参数:
        channels (int): 特征通道数
        seq_len (int): 序列长度
    """
    def __init__(self, channels, seq_len):
        super(TemporalPatternAssembling, self).__init__()
        
        # 全局到局部映射
        self.global_to_local = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 局部到全局映射
        self.local_to_global = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, low_freq, high_freq):
        """
        前向传播
        
        参数:
            low_freq (tensor): 低频分量 [batch_size, channels, seq_len]
            high_freq (tensor): 高频分量 [batch_size, channels, seq_len]
            
        返回:
            tensor: 增强的特征 [batch_size, channels, seq_len]
        """
        # 全局到局部交互
        global_context = self.global_to_local(low_freq)
        enhanced_high_freq = high_freq * global_context
        
        # 局部到全局交互
        local_context = self.local_to_global(high_freq)
        enhanced_low_freq = low_freq * local_context
        
        # 特征融合
        concat_features = torch.cat([enhanced_low_freq, enhanced_high_freq], dim=1)
        fused_features = self.fusion(concat_features)
        
        return fused_features


class MSCNN(nn.Module):
    """
    多尺度卷积神经网络 (Multi-Scale Convolutional Neural Network)
    用于从时间序列中提取多尺度特征，区分短期波动和长期趋势
    
    参数:
        in_channels (int): 输入通道数（变量数）
        base_channels (int): 基础通道数
        ms_blocks (int): 多尺度块的数量
        scales (list): 每个多尺度块使用的卷积核大小列表
        seq_len (int): 输入序列长度
        output_dim (int): 输出特征维度
        use_decoupling (bool): 是否使用时间模式解耦
    """
    def __init__(self, in_channels=1, base_channels=64, ms_blocks=3, scales=[3, 5, 7, 9], 
                 seq_len=96, output_dim=512, use_decoupling=True):
        super(MSCNN, self).__init__()
        
        self.use_decoupling = use_decoupling
        
        # 初始特征提取
        self.init_conv = Conv1dBlock(in_channels, base_channels, kernel_size=3, padding=1)
        
        # 时间模式解耦
        if use_decoupling:
            self.pattern_decoupling = TemporalPatternDecoupling(wavelet='db4', level=1)
            self.pattern_assembling = TemporalPatternAssembling(base_channels, seq_len)
        
        # 多尺度块
        self.ms_layers = nn.ModuleList()
        curr_channels = base_channels
        
        for i in range(ms_blocks):
            out_channels = base_channels * (2 ** i)
            self.ms_layers.append(MSBlock(curr_channels, out_channels, scales))
            curr_channels = out_channels
            
            # 在每个多尺度块后加入下采样操作
            if i < ms_blocks - 1:
                self.ms_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        
        # 全局特征提取
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 投影到指定维度
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(curr_channels, output_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (tensor): 输入时间序列 [batch_size, channels, seq_len]
            
        返回:
            tensor: 提取的特征 [batch_size, output_dim]
        """
        # 初始特征提取
        x = self.init_conv(x)
        
        # 时间模式解耦和组装
        if self.use_decoupling:
            low_freq, high_freq = self.pattern_decoupling(x)
            x = self.pattern_assembling(low_freq, high_freq)
        
        # 多尺度特征提取
        for layer in self.ms_layers:
            x = layer(x)
        
        # 全局池化
        x = self.global_pool(x)
        
        # 投影到指定维度
        x = self.projection(x)
        
        return x


class MSCNNWithAttention(nn.Module):
    """
    带注意力机制的多尺度卷积神经网络
    
    参数:
        in_channels (int): 输入通道数
        base_channels (int): 基础通道数
        ms_blocks (int): 多尺度块的数量
        scales (list): 每个多尺度块使用的卷积核大小列表
        seq_len (int): 输入序列长度
        output_dim (int): 输出特征维度
        use_decoupling (bool): 是否使用时间模式解耦
    """
    def __init__(self, in_channels=1, base_channels=64, ms_blocks=3, scales=[3, 5, 7, 9], 
                 seq_len=96, output_dim=512, use_decoupling=True):
        super(MSCNNWithAttention, self).__init__()
        
        # 基础MSCNN
        self.mscnn = MSCNN(
            in_channels=in_channels,
            base_channels=base_channels,
            ms_blocks=ms_blocks,
            scales=scales,
            seq_len=seq_len,
            output_dim=output_dim,
            use_decoupling=use_decoupling
        )
        
        # 自注意力机制
        self.self_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            batch_first=True
        )
        
        # 最终投影
        self.final_projection = nn.Linear(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (tensor): 输入时间序列 [batch_size, channels, seq_len]
            
        返回:
            tensor: 提取的特征 [batch_size, output_dim]
        """
        # 使用基础MSCNN提取特征
        features = self.mscnn(x)
        
        # 添加维度用于自注意力 [batch_size, 1, output_dim]
        attention_input = features.unsqueeze(1)
        
        # 应用自注意力
        attn_output, _ = self.self_attention(
            attention_input, attention_input, attention_input
        )
        
        # 移除额外维度并应用残差连接
        attn_output = attn_output.squeeze(1) + features
        
        # 层归一化和最终投影
        output = self.layer_norm(attn_output)
        output = self.final_projection(output)
        
        return output


if __name__ == "__main__":
    # 简单测试
    batch_size = 4
    channels = 1  # 单变量时间序列
    seq_len = 96  # 序列长度
    
    # 创建随机输入张量
    x = torch.randn(batch_size, channels, seq_len)
    
    # 测试MSCNN
    model = MSCNN(in_channels=channels, seq_len=seq_len)
    output = model(x)
    logger.info(f"MSCNN 输出形状: {output.shape}")
    
    # 测试带注意力的MSCNN
    model_with_attn = MSCNNWithAttention(in_channels=channels, seq_len=seq_len)
    output_attn = model_with_attn(x)
    logger.info(f"MSCNNWithAttention 输出形状: {output_attn.shape}") 