#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM-PS: 时间序列预测模型的默认配置
"""

import os
import sys
from log.logger import get_logger

# 创建logger
logger = get_logger("config", "llmps_model.log")

def get_default_config():
    """
    获取默认配置
    
    返回:
        dict: 包含所有配置参数的字典
    """
    # 时间序列编码器配置
    temporal_encoder_config = {
        'in_channels': 1,          # 输入通道数（变量数量）
        'base_channels': 64,       # 基础通道数
        'ms_blocks': 3,            # 多尺度块数量
        'scales': [3, 5, 7, 9],    # 卷积核尺寸
        'seq_len': 96,             # 输入序列长度
        'output_dim': 512,         # 输出特征维度
        'use_decoupling': True     # 是否使用时间模式解耦
    }
    
    # 语义编码器配置
    semantic_encoder_config = {
        'in_channels': 1,          # 输入通道数（变量数量）
        'patch_size': 24,          # 时间片段大小
        'overlap': 8,              # 片段重叠大小
        'embed_dim': 96,           # 嵌入维度
        'num_encoder_layers': 4,   # 编码器层数
        'num_decoder_layers': 1,   # 解码器层数
        'nhead': 4,                # 注意力头数
        'dim_feedforward': 384,    # 前馈网络维度
        'dropout': 0.1,            # Dropout比率
        'mask_ratio': 0.75,        # 掩码比率（用于自监督学习）
        'vocab_size': 5000,        # 词汇表大小
        'output_dim': 512,         # 输出特征维度
        'max_prompt_len': 50       # 最大提示词长度
    }
    
    # 融合模块配置
    fusion_config = {
        'temporal_dim': temporal_encoder_config['output_dim'],   # 时间特征维度
        'text_dim': semantic_encoder_config['output_dim'],       # 文本特征维度
        'fusion_dim': 512,         # 融合后的特征维度
        'num_heads': 8,            # 注意力头数
        'dropout': 0.1             # Dropout比率
    }
    
    # 提示词优化器配置
    optimizer_config = {
        'input_dim': fusion_config['fusion_dim'],   # 输入特征维度
        'hidden_dim': 1024,        # 隐藏层维度
        'vocab_size': semantic_encoder_config['vocab_size'],  # 词汇表大小
        'dropout': 0.1             # Dropout比率
    }
    
    # 预测头配置
    forecasting_config = {
        'input_dim': fusion_config['fusion_dim'],   # 输入特征维度
        'hidden_dim': 1024,        # 隐藏层维度
        'forecast_len': 96,        # 预测长度
        'num_variables': temporal_encoder_config['in_channels'],  # 变量数量
        'dropout': 0.1             # Dropout比率
    }
    
    # 数据配置
    data_config = {
        'seq_len': 96,             # 输入序列长度
        'forecast_len': 96,        # 预测长度
        'train_stride': 24,        # 训练数据采样步长
        'val_stride': 96,          # 验证数据采样步长
        'test_stride': 96,         # 测试数据采样步长
        'scale_data': True,        # 是否缩放数据
        'use_time_features': True  # 是否使用时间特征
    }
    
    # 训练配置
    training_config = {
        'batch_size': 64,          # 批次大小
        'epochs': 100,             # 训练轮数
        'learning_rate': 1e-4,     # 学习率
        'weight_decay': 1e-5,      # 权重衰减
        'grad_clip': 1.0,          # 梯度裁剪
        'lr_factor': 0.5,          # 学习率衰减因子
        'lr_patience': 10,         # 学习率衰减耐心值
        'feature_weight': 0.5,     # 特征损失权重 (λ)
        'prompt_weight': 0.1,      # 提示词损失权重 (μ)
        'checkpoint_interval': 5,  # 检查点保存间隔
        'log_interval': 50         # 日志记录间隔
    }
    
    # 评估配置
    evaluation_config = {
        'batch_size': 128,         # 批次大小
        'metrics': ['mse', 'mae', 'rmse', 'mape']  # 评估指标
    }
    
    # 模型配置
    model_config = {
        'temporal_encoder_config': temporal_encoder_config,
        'semantic_encoder_config': semantic_encoder_config,
        'fusion_config': fusion_config,
        'optimizer_config': optimizer_config,
        'forecasting_config': forecasting_config,
        'use_attention': True,     # 是否使用注意力
        'generate_prompts': True   # 是否生成提示词
    }
    
    logger.info("已加载默认配置")
    
    # 返回完整配置
    return {
        'temporal_encoder_config': temporal_encoder_config,
        'semantic_encoder_config': semantic_encoder_config,
        'fusion_config': fusion_config,
        'optimizer_config': optimizer_config,
        'forecasting_config': forecasting_config,
        'data': data_config,
        'training': training_config,
        'evaluation': evaluation_config,
        'use_attention': model_config['use_attention'],
        'generate_prompts': model_config['generate_prompts']
    } 