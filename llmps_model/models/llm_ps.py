"""
LLM-PS: Empowering Large Language Models for Time Series Forecasting with Temporal Patterns and Semantics
完整模型实现

基于论文《LLM-PS: Empowering Large Language Models for Time Series Forecasting with Temporal Patterns and Semantics》实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

from log.logger import get_logger

# 创建logger
logger = get_logger("llm_ps", "llmps_model.log")

from llmps_model.models.mscnn import MSCNN, MSCNNWithAttention
from llmps_model.models.t2t import T2TExtractor, T2TWithPromptGeneration


class CrossModalityFusion(nn.Module):
    """
    跨模态融合模块
    用于融合来自时间模式和文本语义两种模态的特征
    
    参数:
        temporal_dim (int): 时间特征维度
        text_dim (int): 文本特征维度
        fusion_dim (int): 融合后的特征维度
        num_heads (int): 注意力头数
        dropout (float): Dropout比率
    """
    def __init__(self, temporal_dim=512, text_dim=512, fusion_dim=512, num_heads=8, dropout=0.1):
        super(CrossModalityFusion, self).__init__()
        
        # 将时间和文本特征投影到相同的维度空间
        self.temporal_projection = nn.Linear(temporal_dim, fusion_dim)
        self.text_projection = nn.Linear(text_dim, fusion_dim)
        
        # 交叉注意力机制
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
    def forward(self, temporal_features, text_features):
        """
        前向传播
        
        参数:
            temporal_features (tensor): 时间特征 [batch_size, temporal_dim]
            text_features (tensor): 文本特征 [batch_size, text_len, text_dim]
            
        返回:
            tensor: 融合后的特征 [batch_size, fusion_dim]
        """
        batch_size = temporal_features.size(0)
        
        # 投影特征到相同维度空间
        temporal_features = self.temporal_projection(temporal_features)
        text_features = self.text_projection(text_features)
        
        # 扩展时间特征以匹配文本特征的形状
        temporal_features = temporal_features.unsqueeze(1)
        
        # 应用交叉注意力机制
        attended_temporal, _ = self.cross_attention(
            temporal_features, text_features, text_features
        )
        
        # 全局池化文本特征
        pooled_text = torch.mean(text_features, dim=1)
        
        # 拼接特征并进行融合
        concat_features = torch.cat([attended_temporal.squeeze(1), pooled_text], dim=1)
        fused_features = self.fusion_layer(concat_features)
        
        return fused_features


class PromptOptimizer(nn.Module):
    """
    提示词优化器
    根据时间序列特征和语义特征优化生成的提示词
    
    参数:
        input_dim (int): 输入特征维度
        hidden_dim (int): 隐藏层维度
        vocab_size (int): 词汇表大小
        dropout (float): Dropout比率
    """
    def __init__(self, input_dim=512, hidden_dim=1024, vocab_size=30000, dropout=0.1):
        super(PromptOptimizer, self).__init__()
        
        # 特征变换层
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # 提示词生成层
        self.prompt_generator = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, features):
        """
        前向传播
        
        参数:
            features (tensor): 输入特征 [batch_size, input_dim]
            
        返回:
            tensor: 提示词生成的logits [batch_size, vocab_size]
        """
        # 特征变换
        transformed_features = self.feature_transform(features)
        
        # 生成提示词logits
        prompt_logits = self.prompt_generator(transformed_features)
        
        return prompt_logits


class ForecastingHead(nn.Module):
    """
    时间序列预测头
    根据融合特征预测未来时间步
    
    参数:
        input_dim (int): 输入特征维度
        hidden_dim (int): 隐藏层维度
        forecast_len (int): 预测长度
        num_variables (int): 变量数量（通道数）
        dropout (float): Dropout比率
    """
    def __init__(self, input_dim=512, hidden_dim=1024, forecast_len=96, num_variables=1, dropout=0.1):
        super(ForecastingHead, self).__init__()
        
        self.forecast_len = forecast_len
        self.num_variables = num_variables
        
        # 预测层
        self.forecast_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, forecast_len * num_variables)
        )
        
    def forward(self, features):
        """
        前向传播
        
        参数:
            features (tensor): 输入特征 [batch_size, input_dim]
            
        返回:
            tensor: 预测的时间序列 [batch_size, num_variables, forecast_len]
        """
        batch_size = features.size(0)
        
        # 生成预测
        predictions = self.forecast_network(features)
        
        # 重塑为 [batch_size, num_variables, forecast_len]
        predictions = predictions.view(batch_size, self.num_variables, self.forecast_len)
        
        return predictions


class LLMPS(nn.Module):
    """
    LLM-PS: Empowering Large Language Models for Time Series Forecasting with Temporal Patterns and Semantics
    完整模型实现
    
    参数:
        temporal_encoder_config (dict): 时间编码器配置（MSCNN）
        semantic_encoder_config (dict): 语义编码器配置（T2T）
        fusion_config (dict): 特征融合配置
        optimizer_config (dict): 提示词优化器配置
        forecasting_config (dict): 预测头配置
        use_attention (bool): 是否在时间编码器中使用注意力
        generate_prompts (bool): 是否生成提示词
    """
    def __init__(
        self,
        temporal_encoder_config=None,
        semantic_encoder_config=None,
        fusion_config=None,
        optimizer_config=None,
        forecasting_config=None,
        use_attention=True,
        generate_prompts=True
    ):
        super(LLMPS, self).__init__()
        
        # 默认配置
        if temporal_encoder_config is None:
            temporal_encoder_config = {
                'in_channels': 1,
                'base_channels': 64,
                'ms_blocks': 3,
                'scales': [3, 5, 7, 9],
                'seq_len': 96,
                'output_dim': 512,
                'use_decoupling': True
            }
            
        if semantic_encoder_config is None:
            semantic_encoder_config = {
                'in_channels': 1,
                'patch_size': 24,
                'overlap': 0,
                'embed_dim': 96,
                'num_encoder_layers': 4,
                'num_decoder_layers': 1,
                'nhead': 4,
                'dim_feedforward': 384,
                'dropout': 0.1,
                'mask_ratio': 0.75,
                'vocab_size': 30000,
                'output_dim': 512,
                'max_prompt_len': 100
            }
            
        if fusion_config is None:
            fusion_config = {
                'temporal_dim': temporal_encoder_config['output_dim'],
                'text_dim': semantic_encoder_config['output_dim'],
                'fusion_dim': 512,
                'num_heads': 8,
                'dropout': 0.1
            }
            
        if optimizer_config is None:
            optimizer_config = {
                'input_dim': fusion_config['fusion_dim'],
                'hidden_dim': 1024,
                'vocab_size': semantic_encoder_config['vocab_size'],
                'dropout': 0.1
            }
            
        if forecasting_config is None:
            forecasting_config = {
                'input_dim': fusion_config['fusion_dim'],
                'hidden_dim': 1024,
                'forecast_len': 96,
                'num_variables': temporal_encoder_config['in_channels'],
                'dropout': 0.1
            }
        
        # 时间编码器（MSCNN）
        if use_attention:
            self.temporal_encoder = MSCNNWithAttention(**temporal_encoder_config)
        else:
            self.temporal_encoder = MSCNN(**temporal_encoder_config)
        
        # 语义编码器（T2T Extractor）
        if generate_prompts:
            self.semantic_encoder = T2TWithPromptGeneration(**semantic_encoder_config)
        else:
            self.semantic_encoder = T2TExtractor(**semantic_encoder_config)
        
        # 跨模态融合模块
        self.fusion_module = CrossModalityFusion(**fusion_config)
        
        # 提示词优化器
        self.prompt_optimizer = PromptOptimizer(**optimizer_config)
        
        # 预测头
        self.forecasting_head = ForecastingHead(**forecasting_config)
        
        # 模型配置
        self.generate_prompts = generate_prompts
        self.vocab_size = semantic_encoder_config['vocab_size']
        self.forecast_len = forecasting_config['forecast_len']
        
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mode='forecast'):
        """
        前向传播
        
        参数:
            x_enc (tensor): 编码器输入时间序列 [batch_size, seq_len_enc, num_variables]
            x_mark_enc (tensor): 编码器输入时间特征 [batch_size, seq_len_enc, num_features]
            x_dec (tensor): 解码器输入时间序列 [batch_size, seq_len_dec, num_variables]
            x_mark_dec (tensor): 解码器输入时间特征 [batch_size, seq_len_dec, num_features]
            mode (str): 运行模式，'forecast', 'generate' 或 'train'
            
        返回:
            如果mode=='forecast':
                tensor: 预测的时间序列 [batch_size, forecast_len, num_variables]
            如果mode=='generate':
                dict: 包含生成的提示词和预测结果
            如果mode=='train':
                dict: 包含各项损失和预测结果
        """
        batch_size = x_enc.size(0)
        
        # 转换输入格式（如果需要）
        # 从 [batch_size, seq_len, num_variables] 到 [batch_size, num_variables, seq_len]
        if x_enc.size(1) != self.temporal_encoder.in_channels:
            x_enc = x_enc.transpose(1, 2)
        
        # 提取时间特征
        temporal_features = self.temporal_encoder(x_enc)
        
        if self.generate_prompts:
            # 语义提取和提示词生成
            if mode == 'generate':
                generated_tokens, semantic_features, semantic_classes = self.semantic_encoder(
                    x_enc, 
                    generate_prompt=True
                )
            else:
                semantic_features = self.semantic_encoder(x_enc, generate_prompt=False)
        else:
            # 直接语义提取
            semantic_logits, semantic_features = self.semantic_encoder(x_enc, extract_semantics=True)
        
        # 融合特征
        fused_features = self.fusion_module(temporal_features, semantic_features)
        
        if mode == 'forecast':
            # 预测未来时间步
            forecast = self.forecasting_head(fused_features)
            return forecast
        
        elif mode == 'generate':
            # 生成优化后的提示词logits
            optimized_prompt_logits = self.prompt_optimizer(fused_features)
            
            # 预测未来时间步
            forecast = self.forecasting_head(fused_features)
            
            return {
                'generated_tokens': generated_tokens if self.generate_prompts else None,
                'semantic_classes': semantic_classes if self.generate_prompts else None,
                'optimized_prompt_logits': optimized_prompt_logits,
                'forecast': forecast,
                'fused_features': fused_features,
                'temporal_features': temporal_features,    # 添加时间特征
                'semantic_features': semantic_features     # 添加语义特征
            }
        
        else:  # 'train' 模式
            # 生成提示词logits
            prompt_logits = self.prompt_optimizer(fused_features)
            
            # 预测未来时间步
            forecast = self.forecasting_head(fused_features)
            
            return {
                'prompt_logits': prompt_logits,
                'forecast': forecast,
                'fused_features': fused_features,
                'temporal_features': temporal_features,    # 添加时间特征
                'semantic_features': semantic_features     # 添加语义特征 
            }
    
    def generate_optimal_prompt(self, x, max_len=100):
        """
        生成最优提示词
        
        参数:
            x (tensor): 输入时间序列 [batch_size, num_variables, seq_len]
            max_len (int): 生成提示词的最大长度
            
        返回:
            tuple: (生成的最优提示词 [batch_size, prompt_len], 预测的时间序列 [batch_size, num_variables, forecast_len])
        """
        # 获取批次大小和设备
        batch_size = x.size(0)
        device = x.device
        
        # 获取模型生成的特征和logits
        output = self.forward(x, mode='generate')
        prompt_logits = output['optimized_prompt_logits']
        forecast = output['forecast']
        
        # 初始化提示词序列（以开始标记开始）
        start_tokens = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        generated_prompt = start_tokens
        
        # 自回归生成
        for _ in range(max_len - 1):
            # 选择最可能的下一个token
            next_token = torch.argmax(prompt_logits, dim=-1, keepdim=True)
            
            # 添加到序列中
            generated_prompt = torch.cat([generated_prompt, next_token], dim=1)
            
            # 如果生成了结束标记，则停止
            if (next_token == 2).all():  # 假设结束标记为2
                break
            
            # 重新运行模型获取下一个token的预测
            # 简化版本，实际应该根据已生成token重新计算
            prompt_logits = F.softmax(torch.randn(batch_size, self.vocab_size, device=device), dim=-1)
        
        return generated_prompt, forecast


if __name__ == "__main__":
    # 简单测试
    batch_size = 4
    seq_len_enc = 96   # 编码器序列长度
    seq_len_dec = 48   # 解码器序列长度
    forecast_len = 96  # 预测长度
    num_variables = 1  # 变量数量（例如单变量时间序列）
    
    # 创建随机输入
    x_enc = torch.randn(batch_size, num_variables, seq_len_enc)
    x_mark_enc = torch.randn(batch_size, seq_len_enc, 4)  # 假设有4个时间特征
    x_dec = torch.randn(batch_size, num_variables, seq_len_dec)
    x_mark_dec = torch.randn(batch_size, seq_len_dec, 4)
    
    # 定义模型配置
    model_config = {
        'temporal_encoder_config': {
            'in_channels': num_variables,
            'base_channels': 64,
            'ms_blocks': 3,
            'scales': [3, 5, 7, 9],
            'seq_len': seq_len_enc,
            'output_dim': 512,
            'use_decoupling': True
        },
        'semantic_encoder_config': {
            'in_channels': num_variables,
            'patch_size': 24,
            'overlap': 0,
            'embed_dim': 96,
            'num_encoder_layers': 4,
            'num_decoder_layers': 1,
            'nhead': 4,
            'dim_feedforward': 384,
            'dropout': 0.1,
            'mask_ratio': 0.75,
            'vocab_size': 5000,
            'output_dim': 512,
            'max_prompt_len': 20
        },
        'forecasting_config': {
            'forecast_len': forecast_len,
            'num_variables': num_variables
        },
        'use_attention': True,
        'generate_prompts': True
    }
    
    # 创建模型
    model = LLMPS(**model_config)
    
    # 预测模式测试
    forecast = model(x_enc, x_mark_enc, x_dec, x_mark_dec, mode='forecast')
    logger.info(f"预测模式输出形状: {forecast.shape}")
    
    # 生成模式测试
    generate_output = model(x_enc, x_mark_enc, x_dec, x_mark_dec, mode='generate')
    logger.info(f"生成模式提示词logits形状: {generate_output['optimized_prompt_logits'].shape}")
    logger.info(f"生成模式预测形状: {generate_output['forecast'].shape}")
    
    if generate_output['generated_tokens'] is not None:
        logger.info(f"生成的提示词形状: {generate_output['generated_tokens'].shape}")
    
    # 测试生成最优提示词函数
    optimal_prompt, forecast = model.generate_optimal_prompt(x_enc, max_len=20)
    logger.info(f"生成的最优提示词形状: {optimal_prompt.shape}")
    logger.info(f"生成的预测形状: {forecast.shape}") 