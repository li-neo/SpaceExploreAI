"""
T2T Extractor (Time-to-Text Extractor)
用于从时间序列中提取语义表示并生成描述性提示词
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# 添加项目根目录到路径
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from log.logger import get_logger

# 创建logger
logger = get_logger("t2t", "llmps_model.log")


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    为时间片段提供位置信息
    
    参数:
        embed_dim (int): 嵌入维度
        max_len (int): 最大序列长度
        dropout (float): Dropout比率
    """
    def __init__(self, embed_dim, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为缓冲区，不作为模型参数
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """前向传播"""
        # 添加位置编码
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """
    时间片段嵌入
    将时间序列划分为片段进行嵌入
    
    参数:
        in_channels (int): 输入通道数
        patch_size (int): 片段大小
        overlap (int): 片段重叠大小
        embed_dim (int): 嵌入维度
    """
    def __init__(self, in_channels=1, patch_size=24, overlap=0, embed_dim=96):
        super(PatchEmbedding, self).__init__()
        
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap  # 考虑重叠的步长
        
        # 片段的嵌入层
        self.projection = nn.Linear(patch_size * in_channels, embed_dim)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (tensor): 输入时间序列 [batch_size, in_channels, seq_len]
            
        返回:
            tensor: 嵌入后的片段 [batch_size, num_patches, embed_dim]
        """
        batch_size, channels, seq_len = x.shape
        
        # 计算片段数
        num_patches = max(1, (seq_len - self.patch_size) // self.stride + 1)
        
        # 收集片段
        patches = []
        for i in range(num_patches):
            start_idx = i * self.stride
            end_idx = start_idx + self.patch_size
            
            if end_idx <= seq_len:
                # 提取片段
                patch = x[:, :, start_idx:end_idx]
                
                # 重塑为 [batch_size, channels*patch_size]
                patch = patch.reshape(batch_size, -1)
                
                patches.append(patch)
        
        # 堆叠所有片段
        patches = torch.stack(patches, dim=1)  # [batch_size, num_patches, channels*patch_size]
        
        # 嵌入
        patches_embed = self.projection(patches)  # [batch_size, num_patches, embed_dim]
        
        return patches_embed


class TimePatchMasking(nn.Module):
    """
    时间片段掩码
    随机掩码某些时间片段，用于自监督学习
    
    参数:
        mask_ratio (float): 掩码比率
    """
    def __init__(self, mask_ratio=0.75):
        super(TimePatchMasking, self).__init__()
        self.mask_ratio = mask_ratio
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (tensor): 输入片段嵌入 [batch_size, num_patches, embed_dim]
            
        返回:
            tuple: (掩码后的片段, 掩码索引, 原始片段)
        """
        batch_size, num_patches, embed_dim = x.shape
        device = x.device
        
        # 保存原始片段
        x_original = x.clone()
        
        # 计算掩码数量
        num_masked = int(self.mask_ratio * num_patches)
        
        # 对于每个批次，生成不同的掩码模式
        masked_indices = []
        masked_x = x.clone()
        
        for i in range(batch_size):
            # 生成随机掩码索引
            rand_indices = torch.randperm(num_patches, device=device)
            mask_indices = rand_indices[:num_masked]
            masked_indices.append(mask_indices)
            
            # 应用掩码（用零替换）
            masked_x[i, mask_indices] = 0
        
        return masked_x, masked_indices, x_original


class T2TExtractor(nn.Module):
    """
    T2T Extractor (Time-to-Text Extractor)
    从时间序列中提取语义信息
    
    参数:
        in_channels (int): 输入通道数
        patch_size (int): 片段大小
        overlap (int): 片段重叠大小
        embed_dim (int): 嵌入维度
        num_encoder_layers (int): 编码器层数
        num_decoder_layers (int): 解码器层数
        nhead (int): 注意力头数
        dim_feedforward (int): 前馈网络维度
        dropout (float): Dropout比率
        mask_ratio (float): 掩码比率（用于自监督学习）
        vocab_size (int): 词汇表大小
        output_dim (int): 输出特征维度
    """
    def __init__(self, in_channels=1, patch_size=24, overlap=0, embed_dim=96, 
                 num_encoder_layers=4, num_decoder_layers=1, nhead=4, 
                 dim_feedforward=384, dropout=0.1, mask_ratio=0.75, 
                 vocab_size=30000, output_dim=512):
        super(T2TExtractor, self).__init__()
        
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        
        # 片段嵌入
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, overlap, embed_dim)
        
        # 片段掩码
        self.patch_masking = TimePatchMasking(mask_ratio)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Transformer解码器（用于重构）
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # 语义分类头
        self.semantic_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, vocab_size)
        )
        
        # 特征投影
        self.feature_projection = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, extract_semantics=False, reconstruct=False, return_all=False):
        """
        前向传播
        
        参数:
            x (tensor): 输入时间序列 [batch_size, in_channels, seq_len]
            extract_semantics (bool): 是否提取语义
            reconstruct (bool): 是否重构序列
            return_all (bool): 是否返回所有中间结果
            
        返回:
            如果extract_semantics=True:
                tuple: (语义logits, 语义特征)
            如果reconstruct=True:
                tensor: 重构后的序列
            如果return_all=True:
                dict: 包含所有中间结果
        """
        # 将时间序列分解为片段并嵌入
        x_patches = self.patch_embedding(x)
        
        # 应用掩码
        masked_patches, mask_indices, original_patches = self.patch_masking(x_patches)
        
        # 添加位置编码
        pos_patches = self.positional_encoding(masked_patches)
        
        # 通过Transformer编码器
        encoded_patches = self.transformer_encoder(pos_patches)
        
        results = {}
        
        if extract_semantics:
            # 计算序列的全局表示
            global_repr = encoded_patches.mean(dim=1)  # [batch_size, embed_dim]
            
            # 语义分类
            semantic_logits = self.semantic_head(global_repr)  # [batch_size, vocab_size]
            
            # 特征投影
            semantic_features = self.feature_projection(global_repr)  # [batch_size, output_dim]
            
            if return_all:
                results['semantic_logits'] = semantic_logits
                results['semantic_features'] = semantic_features
            else:
                return semantic_logits, semantic_features
        
        if reconstruct:
            # 通过Transformer解码器重构
            # 使用编码器输出作为记忆，使用全零序列作为输入
            batch_size, num_patches, _ = original_patches.shape
            tgt = torch.zeros_like(original_patches)
            
            reconstructed = self.transformer_decoder(tgt, encoded_patches)
            
            if return_all:
                results['reconstructed'] = reconstructed
            else:
                return reconstructed
        
        if return_all:
            results['encoded_patches'] = encoded_patches
            results['masked_patches'] = masked_patches
            results['original_patches'] = original_patches
            return results
        
        # 默认返回编码器输出
        return encoded_patches


class T2TWithPromptGeneration(nn.Module):
    """
    带提示词生成的T2T Extractor
    在基础T2T Extractor上增加提示词生成功能
    
    参数:
        in_channels (int): 输入通道数
        patch_size (int): 片段大小
        overlap (int): 片段重叠大小
        embed_dim (int): 嵌入维度
        num_encoder_layers (int): 编码器层数
        num_decoder_layers (int): 解码器层数
        nhead (int): 注意力头数
        dim_feedforward (int): 前馈网络维度
        dropout (float): Dropout比率
        mask_ratio (float): 掩码比率（用于自监督学习）
        vocab_size (int): 词汇表大小
        output_dim (int): 输出特征维度
        max_prompt_len (int): 最大提示词长度
    """
    def __init__(self, in_channels=1, patch_size=24, overlap=0, embed_dim=96, 
                 num_encoder_layers=4, num_decoder_layers=1, nhead=4, 
                 dim_feedforward=384, dropout=0.1, mask_ratio=0.75, 
                 vocab_size=30000, output_dim=512, max_prompt_len=100):
        super(T2TWithPromptGeneration, self).__init__()
        
        # 基础T2T Extractor
        self.t2t_extractor = T2TExtractor(
            in_channels=in_channels,
            patch_size=patch_size,
            overlap=overlap,
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            mask_ratio=mask_ratio,
            vocab_size=vocab_size,
            output_dim=output_dim
        )
        
        self.max_prompt_len = max_prompt_len
        self.vocab_size = vocab_size
        
        # 提示词生成头
        self.prompt_generation_head = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, vocab_size)
        )
        
    def forward(self, x, generate_prompt=False):
        """
        前向传播
        
        参数:
            x (tensor): 输入时间序列 [batch_size, in_channels, seq_len]
            generate_prompt (bool): 是否生成提示词
            
        返回:
            如果generate_prompt=True:
                tuple: (生成的提示词, 语义特征, 语义类别)
            否则:
                tensor: 语义特征 [batch_size, output_dim]
        """
        # 提取语义
        semantic_logits, semantic_features = self.t2t_extractor(x, extract_semantics=True)
        
        # 获取语义类别
        semantic_classes = torch.argmax(semantic_logits, dim=1)  # [batch_size]
        
        if generate_prompt:
            # 生成提示词
            batch_size = x.size(0)
            device = x.device
            
            # 初始化提示词序列（以开始标记开始）
            start_tokens = torch.ones(batch_size, 1, dtype=torch.long, device=device)
            generated_tokens = start_tokens
            
            # 自回归生成
            for _ in range(self.max_prompt_len - 1):
                # 获取下一个token的logits
                token_logits = self.prompt_generation_head(semantic_features)  # [batch_size, vocab_size]
                
                # 选择最可能的下一个token
                next_token = torch.argmax(token_logits, dim=1, keepdim=True)  # [batch_size, 1]
                
                # 添加到序列中
                generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
                
                # 如果生成了结束标记，则停止
                if (next_token == 2).all():  # 假设结束标记为2
                    break
                
                # 更新特征（简化版本）
                # 在实际应用中，应基于已生成的token重新计算特征
                semantic_features = semantic_features * 0.9
            
            return generated_tokens, semantic_features, semantic_classes
        
        else:
            return semantic_features


if __name__ == "__main__":
    # 简单测试
    batch_size = 4
    in_channels = 1
    seq_len = 96
    
    # 创建随机输入
    x = torch.randn(batch_size, in_channels, seq_len)
    
    # 测试T2TExtractor
    model = T2TExtractor()
    semantic_logits, semantic_features = model(x, extract_semantics=True)
    logger.info(f"语义logits形状: {semantic_logits.shape}")
    logger.info(f"语义特征形状: {semantic_features.shape}")
    
    # 测试重构
    reconstructed = model(x, reconstruct=True)
    logger.info(f"重构输出形状: {reconstructed.shape}")
    
    # 测试T2TWithPromptGeneration
    model_with_prompt = T2TWithPromptGeneration()
    generated_tokens, semantic_features, semantic_classes = model_with_prompt(x, generate_prompt=True)
    logger.info(f"生成的提示词形状: {generated_tokens.shape}")
    logger.info(f"语义特征形状: {semantic_features.shape}")
    logger.info(f"语义类别形状: {semantic_classes.shape}") 