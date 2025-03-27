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
        """
        初始化MSBlock
        
        参数:
            channels (int): 通道数（输入和输出相同）
            num_branches (int): 分支数量，默认为4
            seq_len (int): 序列长度
        """
        super(MSBlock, self).__init__()
        
        # 通道划分数量
        self.num_branches = num_branches
        
        # 每个分支的通道数
        self.branch_channels = channels // num_branches
        
        # 论文公式(1)(2)中的初始1×1卷积，用于特征转换
        self.initial_conv = nn.Conv1d(channels, channels, kernel_size=1)
        
        # 创建每个分支的卷积层，使用不同大小的卷积核以获取多尺度特征
        # 根据论文Fig.3，使用不同尺寸的卷积核来获取不同大小的感受野
        self.branch_convs = nn.ModuleList()
        kernel_sizes = [3, 5, 7, 9]  # 对应不同的感受野大小
        for i in range(num_branches):
            # 确保kernel_size不超过可用的预设值
            k_size = kernel_sizes[i] if i < len(kernel_sizes) else kernel_sizes[-1]
            # 保持输出序列长度不变，调整padding
            padding = k_size // 2
            self.branch_convs.append(
                Conv1dBlock(self.branch_channels, self.branch_channels, kernel_size=k_size, padding=padding)
            )
        
        # 论文公式(3)中的时间模式解耦，使用小波变换提取短期和长期模式
        # 根据序列长度自动计算合适的小波分解级别
        max_level = pywt.dwt_max_level(data_len=seq_len, filter_len=pywt.Wavelet('db4').dec_len)
        self.pattern_decoupling = TemporalPatternDecoupling(
            wavelet='db4', 
            level=min(2, max_level),  # 使用适当的分解级别
            mode='symmetric'
        )
        
        # 用于合并短期和长期模式的1×1卷积，对应论文公式(6)
        self.pattern_fusion = nn.Conv1d(self.branch_channels * 2, self.branch_channels, kernel_size=1)
        
        # 论文公式(2)中的最终融合卷积
        self.fusion_conv = nn.Conv1d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        """
        前向传播 - 实现论文中的多尺度特征提取和模式组装过程
        
        参数:
            x (tensor): 输入特征 [batch_size, channels, seq_len]
            
        返回:
            tensor: 处理后的特征 [batch_size, channels, seq_len]
        """
        # 保存输入用于残差连接 (论文公式(2)中的F_in)
        identity = x
        
        # 初始特征转换
        x = self.initial_conv(x)
        
        # 沿通道维度拆分输入到各个分支
        branch_inputs = torch.split(x, self.branch_channels, dim=1)
        
        # 存储每个分支的短期和长期模式
        branch_outputs = []  # 存储每个分支的原始输出 (论文中的F̄_b)
        branch_short_terms = []  # 存储每个分支的短期模式 P_S^b
        branch_long_terms = []   # 存储每个分支的长期模式 P_L^b
        
        # 对每个分支应用不同尺度的卷积，并解耦为短期和长期模式
        for i, branch_input in enumerate(branch_inputs):
            # 应用分支卷积，获取多尺度特征 F̄_b (对应Fig.3)
            branch_output = self.branch_convs[i](branch_input)
            branch_outputs.append(branch_output)
            
            # 使用小波变换进行时间模式解耦，获取短期模式P_S^b和长期模式P_L^b
            # 对应论文公式(3): W^b_low, {W^b_high_i}^w_i=1 = WT(F̄_b, w)
            short_term, long_term = self.pattern_decoupling(branch_output)
            branch_short_terms.append(short_term)
            branch_long_terms.append(long_term)
        
        # 论文中的时间模式组装过程 (对应公式(4)(5)和Fig.3)
        
        # 短期模式增强: 从低层到高层 (local-to-global)
        # 对应论文公式(4): 从第二个分支开始，累积前一个分支的短期模式
        # For b=2→B do: P_S^b = P_S^b + P_S^(b-1)
        for b in range(1, self.num_branches):
            branch_short_terms[b] = branch_short_terms[b] + branch_short_terms[b-1]
        
        # 长期模式增强: 从高层到低层 (global-to-local)
        # 对应论文公式(5): 从倒数第二个分支开始，累积后一个分支的长期模式
        # For b=(B-1)→1 do: P_L^b = P_L^b + P_L^(b+1)
        for b in range(self.num_branches-2, -1, -1):
            branch_long_terms[b] = branch_long_terms[b] + branch_long_terms[b+1]
        
        # 对应论文公式(6): 合并每个分支的增强模式
        enhanced_branches = []
        for b in range(self.num_branches):
            # 连接短期和长期模式 [batch_size, branch_channels*2, seq_len]
            combined = torch.cat([branch_short_terms[b], branch_long_terms[b]], dim=1)
            # 使用1×1卷积融合模式 [batch_size, branch_channels, seq_len]
            fused = self.pattern_fusion(combined)
            enhanced_branches.append(fused)
        
        # 沿通道维度拼接所有分支 (论文Fig.3中的Concat操作)
        out = torch.cat(enhanced_branches, dim=1)
        
        # 应用最终融合卷积并添加残差连接 (论文公式(2))
        # F_out = Conv_1×1(Concat({F̄_1,...,F̄_B})) + F_in
        out = self.fusion_conv(out)
        out = out + identity
        
        return out


class TemporalPatternDecoupling(nn.Module):
    """
    时间模式解耦模块 - 使用小波变换实现论文公式(3)
    
    使用小波变换将时间序列解耦为高频（短期波动）和低频（长期趋势）组件
    
    参数:
        wavelet (str): 小波类型，默认使用'db4'（DauBechies 4）
        level (int): 分解级别
        mode (str): 边界扩展模式，用于处理信号边界，默认为'symmetric'
    """
    def __init__(self, wavelet='db4', level=2, mode='symmetric'):
        super(TemporalPatternDecoupling, self).__init__()
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        
    def forward(self, x):
        """
        前向传播 - 实现论文公式(3)中的小波变换 WT(F̄_b, w)
        
        参数:
            x (tensor): 输入特征 [batch_size, channels, seq_len]
            
        返回:
            tuple: (P_S^b - 短期模式, P_L^b - 长期模式)
                  P_S^b: 短期模式，来自高频分量
                  P_L^b: 长期模式，来自低频分量
        """
        batch_size, channels, seq_len = x.shape
        device = x.device
        
        # 将张量转换到CPU进行处理 (PyWavelet库要求使用NumPy数组)
        x_cpu = x.detach().cpu()
        
        # 创建用于存储结果的空张量
        long_term = torch.zeros_like(x_cpu)  # 长期模式 (P_L^b)
        short_term = torch.zeros_like(x_cpu)  # 短期模式 (P_S^b)
        
        # 确保小波分解级别不超过信号长度允许的最大级别
        max_level = pywt.dwt_max_level(data_len=seq_len, filter_len=pywt.Wavelet(self.wavelet).dec_len)
        current_level = min(self.level, max_level)
        
        # 对每个批次样本进行处理
        for b in range(batch_size):
            # 将特征reshape为2D张量 [channels, seq_len]
            features_2d = x_cpu[b]
            
            # 创建低频和高频成分的存储
            long_term_components = torch.zeros_like(features_2d)
            short_term_components = torch.zeros_like(features_2d)
            
            # 对每个通道应用小波变换
            for c in range(channels):
                # 提取当前通道数据
                channel_data = features_2d[c].numpy()
                
                try:
                    # 应用小波分解 - 对应论文公式(3)
                    coeffs = pywt.wavedec(channel_data, self.wavelet, level=current_level, mode=self.mode)
                    
                    # 分离低频和高频分量
                    approx_coeffs = coeffs[0]  # 近似系数（低频) - 对应长期模式P_L^b
                    detail_coeffs = coeffs[1:]  # 细节系数（高频) - 对应短期模式P_S^b
                    
                    # 重构低频分量（长期模式）- 保留近似系数，将细节系数设为None
                    low_coeffs = [approx_coeffs] + [None] * len(detail_coeffs)
                    long_term_reconstructed = pywt.waverec(low_coeffs, self.wavelet, mode=self.mode)
                    
                    # 处理重构长度可能与原始长度不匹配的情况
                    if len(long_term_reconstructed) >= seq_len:
                        long_term_components[c] = torch.from_numpy(long_term_reconstructed[:seq_len])
                    else:
                        # 如果重构后长度不足，则填充到原始长度
                        padded = np.pad(long_term_reconstructed, 
                                        (0, seq_len - len(long_term_reconstructed)), 
                                        'constant')
                        long_term_components[c] = torch.from_numpy(padded)
                    
                    # 重构高频分量（短期模式）- 将近似系数设为零，保留细节系数
                    high_coeffs = [np.zeros_like(approx_coeffs)] + detail_coeffs
                    short_term_reconstructed = pywt.waverec(high_coeffs, self.wavelet, mode=self.mode)
                    
                    # 处理重构长度可能与原始长度不匹配的情况
                    if len(short_term_reconstructed) >= seq_len:
                        short_term_components[c] = torch.from_numpy(short_term_reconstructed[:seq_len])
                    else:
                        # 如果重构后长度不足，则填充到原始长度
                        padded = np.pad(short_term_reconstructed, 
                                        (0, seq_len - len(short_term_reconstructed)), 
                                        'constant')
                        short_term_components[c] = torch.from_numpy(padded)
                        
                except Exception as e:
                    # 如果小波变换失败，使用简单的低通和高通滤波作为备选方案
                    logger.warning(f"小波变换失败: {e}，使用备选方案进行模式解耦")
                    # 使用简单的移动平均作为低通滤波器
                    window_size = max(1, seq_len // 8)  # 确保窗口大小至少为1
                    kernel = np.ones(window_size) / window_size
                    # 计算低频分量（移动平均）
                    low_freq = np.convolve(channel_data, kernel, mode='same')
                    # 高频分量 = 原始信号 - 低频分量
                    high_freq = channel_data - low_freq
                    
                    long_term_components[c] = torch.from_numpy(low_freq.astype(np.float32))
                    short_term_components[c] = torch.from_numpy(high_freq.astype(np.float32))
            
            # 存储该批次的结果
            long_term[b] = long_term_components
            short_term[b] = short_term_components
        
        # 将结果移回原始设备
        long_term = long_term.to(device)
        short_term = short_term.to(device)
        
        # 验证重构的有效性: 原始信号应该约等于长期+短期模式
        reconstructed = long_term + short_term
        reconstruction_error = torch.abs(x - reconstructed).max().item()
        if reconstruction_error > 1e-5:
            # 如果误差较大，微调短期模式以减小误差
            short_term = short_term + (x - reconstructed)
        
        # 按照论文中的约定，返回(短期模式, 长期模式) = (P_S^b, P_L^b)
        return short_term, long_term


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
        output_dim (int): 输出特征维度 (注意: 在当前实现中未使用，保留此参数是为了向后兼容)
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
        
        # 记录最终的通道数，这将是输出特征的通道数
        self.final_channels = current_channels
        
        # 注意: 原始投影层在当前实现中不再使用，返回的是[batch_size, seq_len, final_channels]格式的特征
        # self.global_pool = nn.AdaptiveAvgPool1d(1)
        # self.projection = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(current_channels, output_dim),
        #     nn.ReLU(inplace=True)
        # )
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (tensor): 输入时间序列 [batch_size, channels, seq_len]
            
        返回:
            tensor: 提取的多尺度特征 [batch_size, seq_len, channels]
                   其中channels是最后一个MSBlock的通道数(base_channels * 2^(ms_blocks-1))
                   注意：channels与output_dim参数无关，output_dim仅在原设计中用于全局池化后的特征维度
        """
        # 保存输入形状信息
        batch_size, in_channels, seq_len = x.shape
        
        # 初始特征提取
        x = self.init_conv(x)
        
        # 存储中间特征表示
        intermediate_features = []
        
        # 通过所有层处理
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # 在每个MSBlock后保存特征
            if isinstance(layer, MSBlock):
                intermediate_features.append(x)
                
        # 打印当前特征形状，用于调试
        logger.debug(f"MSCNN处理后特征形状: {x.shape}")
        
        # 确保我们至少有一个中间特征
        if not intermediate_features:
            intermediate_features.append(x)
        
        # 选择最后一个MSBlock的输出
        last_features = intermediate_features[-1]  # [batch_size, channels, seq_len']
        
        # 调整特征以匹配输入格式 [batch_size, channels, seq_len'] -> [batch_size, seq_len', features]
        # seq_len'可能小于原始seq_len，因为可能有下采样
        _, channels, curr_seq_len = last_features.shape
        
        # 转置为 [batch_size, seq_len', channels]
        transposed_features = last_features.transpose(1, 2)
        
        # 为了与原始输入的seq_len保持一致，如果需要的话，进行插值
        if curr_seq_len != seq_len:
            # 使用线性插值调整到原始序列长度
            resized_features = F.interpolate(
                transposed_features.transpose(1, 2),  # 转回 [batch_size, channels, seq_len']
                size=seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # 再转回 [batch_size, seq_len, channels]
        else:
            resized_features = transposed_features
        
        return resized_features


class MSCNNWithAttention(nn.Module):
    """
    带注意力机制的多尺度卷积神经网络
    
    参数:
        in_channels (int): 输入通道数
        base_channels (int): 基础通道数
        ms_blocks (int): 多尺度块的数量
        num_branches (int): 每个多尺度块中的分支数量
        seq_len (int): 输入序列长度
        output_dim (int): 最终输出特征维度，用于最后的投影层
        num_heads (int): 注意力头数量
    """
    def __init__(self, in_channels=64, base_channels=128, ms_blocks=3, num_branches=4, 
                 seq_len=96, output_dim=512, num_heads=8):
        super(MSCNNWithAttention, self).__init__()
        
        # 计算最终通道数，这将是MSCNN最后一层输出的通道数
        # 在MSCNN中，通道数在每个MSBlock后翻倍
        final_channels = base_channels * (2 ** (ms_blocks - 1))
        
        # 基础MSCNN，使用更新后的结构，其中每个MSBlock内部具有时间模式解耦和组装功能
        # 注意：MSCNN将返回[batch_size, seq_len, final_channels]格式的特征
        self.mscnn = MSCNN(
            in_channels=in_channels,
            base_channels=base_channels,
            ms_blocks=ms_blocks,
            num_branches=num_branches,
            seq_len=seq_len,
            output_dim=output_dim  # 这个参数在MSCNN的实现中不再使用
        )
        
        # 自注意力机制 - 必须匹配MSCNN实际输出的通道数/特征维度
        # 由于MSCNN返回的是[batch_size, seq_len, final_channels]格式，embed_dim应该是final_channels
        self.self_attention = nn.MultiheadAttention(
            embed_dim=final_channels,  # 使用计算得到的final_channels作为embed_dim
            num_heads=num_heads,
            batch_first=True
        )
        
        # 最终投影 - 将final_channels投影到指定的output_dim
        self.final_projection = nn.Linear(final_channels, output_dim)
        # 确保层归一化维度匹配
        self.layer_norm = nn.LayerNorm(final_channels)  # 使用LayerNorm替代RMSNorm，更为标准
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (tensor): 输入时间序列 [batch_size, channels, seq_len]
            
        返回:
            tensor: 提取的特征 [batch_size, seq_len, output_dim]
                   通过注意力机制和投影层处理后的最终序列特征
        """
        # 使用基础MSCNN提取特征
        features = self.mscnn(x)  # [batch_size, seq_len, channels]
        
        # 打印特征形状以便调试
        logger.info(f"MSCNN输出特征形状: {features.shape}")
        
        # 应用自注意力，不需要额外的维度调整，因为特征已是[batch_size, seq_len, features]格式
        attn_output, _ = self.self_attention(
            features, features, features
        )
        
        # 应用残差连接
        attn_output = attn_output + features
        
        # 层归一化和最终投影
        # 对每个时间步应用层归一化和投影
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
    output_dim = 64  # 输出特征维度，在MSCNNWithAttention中使用，但在MSCNN中未使用
    
    # 创建随机输入张量
    x = torch.randn(batch_size, channels, seq_len)
    
    try:
        # 测试MSCNN
        logger.info("测试MSCNN模型:")
        model = MSCNN(
            in_channels=channels,
            base_channels=base_channels,
            ms_blocks=ms_blocks, 
            num_branches=num_branches,
            seq_len=seq_len,
            output_dim=output_dim  # 此参数在当前实现中未使用
        )
        
        # 计算最终通道数，用于验证
        expected_channels = base_channels * (2 ** (ms_blocks - 1))
        logger.info(f"预期的最终通道数: {expected_channels}")
        
        # 运行模型
        output = model(x)
        logger.info(f"MSCNN 输出形状: {output.shape}")  # 应该是 [batch_size, seq_len, expected_channels]
        
        # 验证输出形状
        assert output.shape == (batch_size, seq_len, expected_channels), \
            f"MSCNN输出形状错误，期望{(batch_size, seq_len, expected_channels)}，得到{output.shape}"
        
        # 测试带注意力的MSCNN
        logger.info("测试MSCNNWithAttention模型:")
        model_with_attn = MSCNNWithAttention(
            in_channels=channels,
            base_channels=base_channels,
            ms_blocks=ms_blocks, 
            num_branches=num_branches,
            seq_len=seq_len,
            output_dim=output_dim,  # 此参数用于最终的特征投影
            num_heads=8
        )
        
        # 运行带注意力的模型
        output_attn = model_with_attn(x)
        logger.info(f"MSCNNWithAttention 输出形状: {output_attn.shape}")  # 应该是 [batch_size, seq_len, output_dim]
        
        # 验证输出形状
        assert output_attn.shape == (batch_size, seq_len, output_dim), \
            f"MSCNNWithAttention输出形状错误，期望{(batch_size, seq_len, output_dim)}，得到{output_attn.shape}"
        
        logger.info("测试完成: 所有形状验证通过!")
    except Exception as e:
        logger.error(f"测试时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())

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