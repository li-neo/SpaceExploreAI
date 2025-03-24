#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM-PS与SpaceExploreAI集成模块
整合时间序列预测特性到SpaceExploreAI框架中
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from log.logger import get_logger

# 创建logger
logger = get_logger("integration", "llmps_model.log")

# 导入SpaceExploreAI的Transformer模型
try:
    from model.transformer import Transformer, TransformerConfig
    from model.attention import Attention
    logger.info("成功导入SpaceExploreAI的Transformer组件")
except ImportError:
    logger.warning("无法导入SpaceExploreAI的Transformer组件，将使用兼容模式运行")
    Transformer = None
    TransformerConfig = None
    Attention = None

# 导入LLM-PS组件
from mscnn import MSCNN, MSCNNWithAttention, TemporalPatternDecoupling
from t2t import T2TExtractor, T2TWithPromptGeneration
from llm_ps import CrossModalityFusion, PromptOptimizer, ForecastingHead


class IntegratedLLMPS(nn.Module):
    """
    集成的LLM-PS模型
    将时间序列预测功能与SpaceExploreAI的LLM模型集成
    
    参数:
        llm_config (dict): LLM配置
        temporal_encoder_config (dict): 时间编码器配置
        semantic_encoder_config (dict): 语义编码器配置
        fusion_config (dict): 融合模块配置
        optimizer_config (dict): 提示词优化器配置
        forecasting_config (dict): 预测头配置
        use_external_llm (bool): 是否使用外部LLM模型
        llm_checkpoint_path (str): 外部LLM模型检查点路径
        use_attention (bool): 是否在时间编码器中使用注意力
    """
    def __init__(
        self,
        llm_config=None,
        temporal_encoder_config=None,
        semantic_encoder_config=None,
        fusion_config=None,
        optimizer_config=None,
        forecasting_config=None,
        use_external_llm=False,
        llm_checkpoint_path=None,
        use_attention=True
    ):
        super(IntegratedLLMPS, self).__init__()
        
        # 默认LLM配置
        if llm_config is None:
            llm_config = {
                'vocab_size': 32000,
                'd_model': 768,
                'n_heads': 12,
                'n_layers': 6,
                'dim_feedforward': 3072,
                'dropout': 0.1,
                'max_seq_len': 1024
            }
            
        # 默认时间编码器配置
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
            
        # 默认语义编码器配置
        if semantic_encoder_config is None:
            semantic_encoder_config = {
                'in_channels': 1,
                'patch_size': 24,
                'overlap': 8,
                'embed_dim': 96,
                'num_encoder_layers': 4,
                'num_decoder_layers': 1,
                'nhead': 4,
                'dim_feedforward': 384,
                'dropout': 0.1,
                'mask_ratio': 0.75,
                'vocab_size': llm_config['vocab_size'],
                'output_dim': 512,
                'max_prompt_len': 50
            }
            
        # 默认融合模块配置
        if fusion_config is None:
            fusion_config = {
                'temporal_dim': temporal_encoder_config['output_dim'],
                'text_dim': semantic_encoder_config['output_dim'],
                'fusion_dim': 512,
                'num_heads': 8,
                'dropout': 0.1
            }
            
        # 默认提示词优化器配置
        if optimizer_config is None:
            optimizer_config = {
                'input_dim': fusion_config['fusion_dim'],
                'hidden_dim': 1024,
                'vocab_size': llm_config['vocab_size'],
                'dropout': 0.1
            }
            
        # 默认预测头配置
        if forecasting_config is None:
            forecasting_config = {
                'input_dim': fusion_config['fusion_dim'],
                'hidden_dim': 1024,
                'forecast_len': 96,
                'num_variables': temporal_encoder_config['in_channels'],
                'dropout': 0.1
            }
        
        # 初始化组件
        # 时间编码器（MSCNN）
        if use_attention:
            self.temporal_encoder = MSCNNWithAttention(**temporal_encoder_config)
        else:
            self.temporal_encoder = MSCNN(**temporal_encoder_config)
        
        # 语义编码器（T2T Extractor）
        self.semantic_encoder = T2TExtractor(**semantic_encoder_config)
        
        # 跨模态融合模块
        self.fusion_module = CrossModalityFusion(**fusion_config)
        
        # 提示词优化器
        self.prompt_optimizer = PromptOptimizer(**optimizer_config)
        
        # 预测头
        self.forecasting_head = ForecastingHead(**forecasting_config)
        
        # 初始化或加载LLM
        self.use_external_llm = use_external_llm
        
        if Transformer is not None and not use_external_llm:
            # 使用SpaceExploreAI的Transformer
            transformer_config = TransformerConfig(
                vocab_size=llm_config['vocab_size'],
                d_model=llm_config['d_model'],
                n_heads=llm_config['n_heads'],
                n_layers=llm_config['n_layers'],
                dim_feedforward=llm_config['dim_feedforward'],
                dropout=llm_config['dropout'],
                max_seq_len=llm_config['max_seq_len']
            )
            self.llm = Transformer(transformer_config)
            logger.info("使用SpaceExploreAI的Transformer模型")
        elif llm_checkpoint_path and os.path.exists(llm_checkpoint_path):
            # 加载预训练的LLM模型
            self.llm = self._load_external_llm(llm_checkpoint_path)
        else:
            # 创建一个简单的兼容模型
            self.llm = self._create_compatible_llm(llm_config)
            logger.info("使用兼容的Transformer模型")
        
        # 保存配置
        self.llm_config = llm_config
        self.temporal_encoder_config = temporal_encoder_config
        self.semantic_encoder_config = semantic_encoder_config
        self.fusion_config = fusion_config
        self.optimizer_config = optimizer_config
        self.forecasting_config = forecasting_config
        
    def _load_external_llm(self, checkpoint_path):
        """
        加载外部预训练的LLM模型
        
        参数:
            checkpoint_path (str): 模型检查点路径
            
        返回:
            nn.Module: 加载的LLM模型
        """
        try:
            # 尝试加载模型
            model = torch.load(checkpoint_path, map_location='cpu')
            logger.info(f"成功加载LLM模型: {checkpoint_path}")
            return model
        except Exception as e:
            logger.error(f"加载LLM模型时出错: {e}")
            logger.warning("将使用兼容模式运行")
            return self._create_compatible_llm(self.llm_config)
            
    def _create_compatible_llm(self, config):
        """
        创建一个简单的兼容LLM模型
        
        参数:
            config (dict): 模型配置
            
        返回:
            nn.Module: 简单的兼容模型
        """
        # 简单的Transformer编码器
        class SimpleTransformer(nn.Module):
            def __init__(self, config):
                super(SimpleTransformer, self).__init__()
                self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=config['d_model'],
                    nhead=config['n_heads'],
                    dim_feedforward=config['dim_feedforward'],
                    dropout=config['dropout'],
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, config['n_layers'])
                self.output_layer = nn.Linear(config['d_model'], config['vocab_size'])
                
            def forward(self, input_ids, attention_mask=None):
                # 将输入token转换为嵌入
                x = self.embedding(input_ids)
                
                # 如果有注意力掩码，应用掩码
                if attention_mask is not None:
                    # 将0-1掩码转换为布尔掩码，并确保形状正确
                    mask = (1 - attention_mask).bool()
                else:
                    mask = None
                
                # 通过Transformer编码器
                hidden_states = self.transformer(x, src_key_padding_mask=mask)
                
                # 输出层
                logits = self.output_layer(hidden_states)
                
                return {
                    'logits': logits,
                    'hidden_states': hidden_states,
                    'embeddings': x
                }
                
        return SimpleTransformer(config)
    
    def forward(self, time_series_input, time_features=None, llm_input_ids=None, attention_mask=None, mode='integrated'):
        """
        前向传播
        
        参数:
            time_series_input (tensor): 时间序列输入 [batch_size, num_variables, seq_len] 或 
                                         [batch_size, seq_len, num_variables]
            time_features (tensor, 可选): 时间特征 [batch_size, seq_len, feature_dim]
            llm_input_ids (tensor, 可选): LLM的输入token IDs [batch_size, seq_len]
            attention_mask (tensor, 可选): 注意力掩码 [batch_size, seq_len]
            mode (str): 运行模式，'integrated'(集成), 'forecast'(预测), 'llm'(仅LLM), 或 'train'(训练)
            
        返回:
            dict: 包含各种输出的字典
        """
        batch_size = time_series_input.size(0)
        
        # 确保时间序列输入格式正确 [batch_size, num_variables, seq_len]
        if time_series_input.size(1) != self.temporal_encoder_config['in_channels']:
            time_series_input = time_series_input.transpose(1, 2)
        
        # 时间序列预测部分
        # 提取时间特征
        temporal_features = self.temporal_encoder(time_series_input)
        
        # 语义特征提取
        if mode in ['integrated', 'train']:
            semantic_logits, semantic_features = self.semantic_encoder(time_series_input, extract_semantics=True)
        else:
            semantic_features = None
        
        # 特征融合（如果有语义特征）
        if semantic_features is not None:
            fused_features = self.fusion_module(temporal_features, semantic_features)
        else:
            fused_features = temporal_features
        
        # 时间序列预测
        forecast = self.forecasting_head(fused_features)
        
        # 提示词生成
        if mode in ['integrated', 'train']:
            prompt_logits = self.prompt_optimizer(fused_features)
            
            # 转换为token索引
            if mode == 'integrated':
                prompt_tokens = torch.argmax(prompt_logits, dim=1)
                
                # 如果未提供LLM输入，则使用生成的提示词
                if llm_input_ids is None:
                    llm_input_ids = prompt_tokens.unsqueeze(1)  # [batch_size, 1]
        else:
            prompt_logits = None
        
        # LLM处理
        llm_outputs = None
        if mode in ['integrated', 'llm'] and llm_input_ids is not None:
            llm_outputs = self.llm(llm_input_ids, attention_mask)
        
        # 返回不同模式的输出
        if mode == 'forecast':
            return forecast
        
        elif mode == 'llm':
            return llm_outputs
        
        elif mode == 'integrated':
            # 集成模式：返回预测结果和LLM输出
            return {
                'forecast': forecast,
                'prompt_logits': prompt_logits,
                'prompt_tokens': prompt_tokens if prompt_logits is not None else None,
                'llm_outputs': llm_outputs
            }
        
        else:  # 'train' 模式
            # 训练模式：返回所有中间特征和输出
            return {
                'forecast': forecast,
                'prompt_logits': prompt_logits,
                'temporal_features': temporal_features,
                'semantic_features': semantic_features,
                'fused_features': fused_features,
                'llm_outputs': llm_outputs
            }
    
    def generate_forecast_with_explanation(self, time_series_input, time_features=None, max_len=100, temperature=0.7):
        """
        生成时间序列预测并附带文本解释
        
        参数:
            time_series_input (tensor): 时间序列输入 [batch_size, num_variables, seq_len] 或 
                                         [batch_size, seq_len, num_variables]
            time_features (tensor, 可选): 时间特征 [batch_size, seq_len, feature_dim]
            max_len (int): 生成文本的最大长度
            temperature (float): 采样温度
            
        返回:
            tuple: (预测结果, 生成的解释文本)
        """
        # 确保时间序列输入格式正确
        if time_series_input.size(1) != self.temporal_encoder_config['in_channels']:
            time_series_input = time_series_input.transpose(1, 2)
        
        # 时间序列预测
        forecast = self.forward(time_series_input, time_features, mode='forecast')
        
        # 生成提示词
        outputs = self.forward(time_series_input, time_features, mode='train')
        prompt_logits = outputs['prompt_logits']
        
        # 将提示词转换为token
        prompt_tokens = torch.argmax(prompt_logits, dim=1).unsqueeze(1)  # [batch_size, 1]
        
        # 使用LLM生成解释
        batch_size = time_series_input.size(0)
        device = time_series_input.device
        
        # 生成输入：提示词作为起始
        input_ids = prompt_tokens
        
        # 自回归生成
        for _ in range(max_len - 1):
            # 当前生成的token
            llm_outputs = self.llm(input_ids, attention_mask=None)
            
            # 获取最后一个token的logits
            next_token_logits = llm_outputs['logits'][:, -1, :]
            
            # 应用温度
            next_token_logits = next_token_logits / temperature
            
            # 采样下一个token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 添加到已生成序列
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # 检查是否所有序列都生成了结束标记（假设结束标记ID为2）
            if (next_token == 2).all():
                break
        
        # 返回结果
        return forecast, input_ids


# 用于测试的简单示例
if __name__ == "__main__":
    # 创建模型
    model = IntegratedLLMPS()
    
    # 创建示例输入
    batch_size = 2
    num_variables = 1
    seq_len = 96
    
    time_series = torch.randn(batch_size, num_variables, seq_len)
    time_features = torch.randn(batch_size, seq_len, 8)  # 8个时间特征
    
    # 测试不同模式
    forecast = model(time_series, time_features, mode='forecast')
    logger.info(f"预测输出形状: {forecast.shape}")
    
    train_outputs = model(time_series, time_features, mode='train')
    logger.info(f"训练模式输出:")
    for k, v in train_outputs.items():
        if isinstance(v, torch.Tensor):
            logger.info(f"  {k}: {v.shape}")
        elif v is None:
            logger.info(f"  {k}: None")
        else:
            logger.info(f"  {k}: {type(v)}")
    
    # 模拟LLM输入
    llm_input_ids = torch.randint(0, 1000, (batch_size, 10))  # [batch_size, seq_len]
    attention_mask = torch.ones_like(llm_input_ids)
    
    integrated_outputs = model(time_series, time_features, llm_input_ids, attention_mask, mode='integrated')
    logger.info(f"集成模式输出:")
    for k, v in integrated_outputs.items():
        if isinstance(v, torch.Tensor):
            logger.info(f"  {k}: {v.shape}")
        elif isinstance(v, dict):
            logger.info(f"  {k}:")
            for k2, v2 in v.items():
                if isinstance(v2, torch.Tensor):
                    logger.info(f"    {k2}: {v2.shape}")
                else:
                    logger.info(f"    {k2}: {type(v2)}")
        elif v is None:
            logger.info(f"  {k}: None")
        else:
            logger.info(f"  {k}: {type(v)}")
    
    # 测试预测并生成解释
    forecast, explanation = model.generate_forecast_with_explanation(time_series, time_features)
    logger.info(f"预测形状: {forecast.shape}")
    logger.info(f"解释token形状: {explanation.shape}") 