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
        
        # Conv1d 卷积操作,主要功能是特征转换，从原始数据中提取有用的模式和表示
        # new_length = ⌊(length + 2padding - dilation(kernel_size-1) - 1)/stride + 1⌋
        # new_length = (32 + 2 * 1 - 2) / 1 = 32
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
        # 输入数据形状: [batch_size,in_channels, seq_len]
        # 卷积操作后输出数据形状: [batch_size, out_channels, seq_len]
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.relu(x)
        return x


class MSBlock(nn.Module):
    """
    多尺度块 (Multi-Scale Block)，严格按照LLM-PS论文实现
    
    参数:
        channels (int): 通道数（输入和输出相同）
        num_branches (int): 分支数量，默认为4
        seq_len (int): 序列长度，用于时间模式组装
    """
    def __init__(self, channels, num_branches=4, seq_len=96):
        super(MSBlock, self).__init__()
        
        # 1×1 卷积用于初始特征转换
        self.initial_conv = nn.Conv1d(channels, channels, kernel_size=1)
        
        # 通道划分数量
        self.num_branches = num_branches
        
        # 创建B个3×3卷积分支
        self.branch_convs = nn.ModuleList()
        for _ in range(num_branches):
            # Conv1dBlock 卷积块： 每个分支都是一个Conv1dBlock实例，这是一个封装了一维卷积、批归一化和ReLU激活的自定义模块
            # 每个分支的输出通道数为 out_channels // num_branches；例如，如果out_channels=128且num_branches=4，则每个分支的输入和输出通道数都是32
            # 每个分支的卷积核大小为3，填充为1
            # Channel-wise Division（通道划分）：将输入特征在通道维度上划分为num_branches份，
            # 每个分支的特征形状为 [batch_size, out_channels // num_branches, seq_len]
            # 递归调用时，分支都会接受前一个分支的输出作为额外输入
            # in & out 都是 out_channels // num_branches，递归的时候可以直接相加，最后在out_channels维度上拼接,完成特征融合
            self.branch_convs.append(
                Conv1dBlock(channels // num_branches, channels // num_branches, 
                           kernel_size=3, padding=1)
            )
        
        # 1×1 卷积用于融合特征
        self.fusion_conv = nn.Conv1d(channels, channels, kernel_size=1)
        
        # 添加时间模式解耦与组装模块
        self.pattern_decoupling = TemporalPatternDecoupling(wavelet='db4', level=1)
        self.pattern_assembling = TemporalPatternAssembling(channels, seq_len)
        
    def channel_division(self, x):
        """
        将特征在通道维度上划分为num_branches份
        
        参数:
            x (tensor): 输入特征 [batch_size, channels, seq_len]
            
        返回:
            list: 划分后的特征列表
        """
        batch_size, channels, seq_len = x.shape
        branch_channels = channels // self.num_branches
        
        divided_features = []
        for i in range(self.num_branches):
            start_idx = i * branch_channels
            end_idx = (i + 1) * branch_channels
            divided_features.append(x[:, start_idx:end_idx, :])
            
        return divided_features
        
    def forward(self, x):
        """前向传播"""
        # 保存输入用于残差连接
        identity = x
        
        # 初始1×1卷积特征转换
        features = self.initial_conv(x)
        
        # 将特征在通道维度上划分为num_branches份
        divided_features = self.channel_division(features)
        
        # 递归地应用卷积块并与前一个分支的输出相加
        branch_outputs = []
        prev_output = None
        
        for i, (branch_feature, branch_conv) in enumerate(zip(divided_features, self.branch_convs)):
            if i == 0:
                # 第一个分支直接应用卷积
                branch_output = branch_conv(branch_feature)
            else:
                # 其他分支将前一个分支的输出与当前特征相加后应用卷积
                branch_output = branch_conv(branch_feature + prev_output)
            
            branch_outputs.append(branch_output)
            prev_output = branch_output
            
        # 拼接所有分支的输出
        concat_features = torch.cat(branch_outputs, dim=1)
        
        # 最后添加残差连接
        # concat_features = concat_features + features
        
        # 为拼接后的特征进行时间模式解耦
        long_term_pattern, short_term_pattern = self.pattern_decoupling(concat_features)
        
        # 时间模式组装，增强时间特征
        assembled_features = self.pattern_assembling(long_term_pattern, short_term_pattern)
        
        # 使用1×1卷积融合特征
        output = self.fusion_conv(assembled_features)
        
        # 添加全局残差连接（输入和输出通道数相同，可以直接相加）
        output = output + identity
        
        return output


class TemporalPatternDecoupling(nn.Module):
    """
    时间模式解耦模块
    使用小波变换将时间序列解耦为高频（短期波动）和低频（长期趋势）组件
    
    参数:
        wavelet (str): 小波类型，默认使用'db4'（DauBechies 4）
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
            x (tensor): 输入特征 [batch_size, channels, seq_len]
            
        返回:
            tuple: (长期模式-低频分量, 短期模式-高频分量)
        """
        batch_size, channels, seq_len = x.shape
        device = x.device
        
        # 将张量转为CPU上的NumPy数组进行小波变换
        x_np = x.detach().cpu().numpy()
        
        # 初始化结果
        low_freq = np.zeros_like(x_np)  # 长期模式
        high_freq = np.zeros_like(x_np)  # 短期模式
        
        # 对每个样本和通道分别进行小波变换
        for b in range(batch_size):
            for c in range(channels):
                # 执行小波分解
                coeffs = pywt.wavedec(x_np[b, c], self.wavelet, level=self.level)
                
                # 分离低频和高频分量
                approx_coeffs = coeffs[0]  # 近似系数（低频）
                detail_coeffs = coeffs[1:]  # 细节系数（高频）
                
                # 重构低频分量（长期模式）
                # 保留近似系数，将细节系数设为0
                low_coeffs = [approx_coeffs] + [None] * len(detail_coeffs)
                low_freq[b, c] = pywt.waverec(low_coeffs, self.wavelet)[:seq_len]
                
                # 重构高频分量（短期模式）
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
        low_freq_tensor = torch.from_numpy(low_freq).to(device)  # 长期模式
        high_freq_tensor = torch.from_numpy(high_freq).to(device)  # 短期模式
        
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
        
        # 全局到局部映射（用于增强短期模式）
        self.global_to_local = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 局部到全局映射（用于增强长期模式）
        self.local_to_global = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 全局池化获取全局上下文
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Sigmoid()  # 生成注意力权重
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, long_term_pattern, short_term_pattern):
        """
        前向传播
        
        参数:
            long_term_pattern (tensor): 长期模式（低频分量）[batch_size, channels, seq_len]
            short_term_pattern (tensor): 短期模式（高频分量）[batch_size, channels, seq_len]
            
        返回:
            tensor: 增强的特征 [batch_size, channels, seq_len]
        """
        # 全局到局部交互：使用长期模式增强短期模式
        global_context = self.global_to_local(long_term_pattern)
        enhanced_short_term = short_term_pattern * global_context
        
        # 局部到全局交互：使用短期模式增强长期模式
        local_context = self.local_to_global(short_term_pattern)
        enhanced_long_term = long_term_pattern * local_context
        
        # 特征融合
        concat_features = torch.cat([enhanced_long_term, enhanced_short_term], dim=1)
        assembled_features = self.fusion(concat_features)
        
        return assembled_features


class MSCNN(nn.Module):
    """
    多尺度卷积神经网络 (Multi-Scale Convolutional Neural Network)
    用于从时间序列中提取多尺度特征，捕获短期波动和长期趋势
    
    参数:
        in_channels (int): 输入通道数（变量数）
        base_channels (int): 基础通道数
        ms_blocks (int): 多尺度块的数量
        num_branches (int): 每个多尺度块中的分支数量
        seq_len (int): 输入序列长度
        output_dim (int): 输出特征维度
    """
    def __init__(self, in_channels=64, base_channels=128, ms_blocks=3, num_branches=4, 
                 seq_len=96, output_dim=512):
        super(MSCNN, self).__init__()
        
        # 1.初始特征提取，从输入通道数转换到基础通道数
        self.init_conv = Conv1dBlock(in_channels, base_channels, kernel_size=3, padding=1)
        
        # 堆叠的多尺度块和通道调整层
        self.layers = nn.ModuleList()
        current_channels = base_channels
        current_seq_len = seq_len
        
        for i in range(ms_blocks):
            # 如果需要增加通道数，先添加通道调整层
            if i > 0:
                next_channels = base_channels * (2 ** i)
                self.layers.append(
                    nn.Sequential(
                        nn.Conv1d(current_channels, next_channels, kernel_size=1),
                        nn.BatchNorm1d(next_channels),
                        nn.ReLU(inplace=True)
                    )
                )
                current_channels = next_channels
            
            # 添加MSBlock，保持输入输出通道数相同
            self.layers.append(MSBlock(current_channels, num_branches, current_seq_len))
            
            # 在每个多尺度块后添加下采样操作（除了最后一个块）
            if i < ms_blocks - 1:
                self.layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
                current_seq_len = current_seq_len // 2
                
        # 全局特征提取
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 投影到指定维度
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(current_channels, output_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (tensor): 输入时间序列 [batch_size, channels, seq_len]
            
        返回:
            tensor: 提取的多尺度特征 [batch_size, output_dim]
        """
        # 初始特征提取
        x = self.init_conv(x)
        
        # 通过所有层处理
        for layer in self.layers:
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
        num_branches (int): 每个多尺度块中的分支数量
        seq_len (int): 输入序列长度
        output_dim (int): 输出特征维度
        num_heads (int): 注意力头数量
    """
    def __init__(self, in_channels=64, base_channels=128, ms_blocks=3, num_branches=4, 
                 seq_len=96, output_dim=512, num_heads=8):
        super(MSCNNWithAttention, self).__init__()
        
        # 基础MSCNN，使用更新后的结构，其中每个MSBlock内部具有时间模式解耦和组装功能
        self.mscnn = MSCNN(
            in_channels=in_channels,
            base_channels=base_channels,
            ms_blocks=ms_blocks,
            num_branches=num_branches,
            seq_len=seq_len,
            output_dim=output_dim
        )
        
        # 自注意力机制
        self.self_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 最终投影
        self.final_projection = nn.Linear(output_dim, output_dim)
        self.layer_norm = nn.RMSNorm(output_dim)
        
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
    batch_size = 16  # 批量大小
    channels = 64  # 输入变量数，代表着时序数据的变量数，如open、close、high、low、volume等
    seq_len = 32  # 序列长度，代表着时序数据的长度，如96个数据点
    base_channels = 128  # 基础通道数，代表着每个多尺度块的输出通道数
    ms_blocks = 3  # 多尺度块数量
    num_branches = 4  # 每个多尺度块中的分支数量
    output_dim = 64  # 输出特征维度，代表着最终输出的特征维度，输出要和SPAI模型输入维度一致   
    
    # 创建随机输入张量
    x = torch.randn(batch_size, channels, seq_len)
    
    # 测试MSCNN
    model = MSCNN(
        in_channels=channels,
        base_channels=base_channels,
        ms_blocks=ms_blocks, 
        num_branches=num_branches,
        seq_len=seq_len,
        output_dim=output_dim
    )
    output = model(x)
    logger.info(f"MSCNN 输出形状: {output.shape}")
    
    # 测试带注意力的MSCNN
    model_with_attn = MSCNNWithAttention(
        in_channels=channels,
        base_channels=base_channels,
        ms_blocks=ms_blocks, 
        num_branches=num_branches,
        seq_len=seq_len,
        output_dim=output_dim,
        num_heads=8
    )
    output_attn = model_with_attn(x)
    logger.info(f"MSCNNWithAttention 输出形状: {output_attn.shape}") 

"""
数据流转换：
输入：[batch_size, channels(64), seq_len(96)]
初始转换后：[batch_size, base_channels(128), seq_len(96)]
第二个MSBlock前：[batch_size, base_channels*2(256), seq_len/2(48)]
第三个MSBlock前：[batch_size, base_channels*4(512), seq_len/4(24)]

MSCNN中的设计体现了一种空间-通道平衡原则：
序列长度减半：seq_len → seq_len/2 → seq_len/4 → ...
通道数翻倍：base_channels → base_channels*2 → base_channels*4 → ...

图像: [batch_size, channels, height, width]
时序: [batch_size, channels, seq_len]
"""