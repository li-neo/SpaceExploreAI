#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM-PS与SpaceExploreAI集成训练脚本
整合训练时间序列预测模型与LLM
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from log.logger import get_logger

# 创建logger
logger = get_logger("train_integrated", "llmps_model.log")

from models.integration import IntegratedLLMPS
from data.dataset import TimeSeriesDataset
from utils.metrics import compute_metrics
from llmps_model.utils.visualization import plot_predictions
from configs.default_config import get_default_config


def save_model(model, optimizer, epoch, config, metrics, save_path):
    """保存模型检查点"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'config': config,
        'metrics': metrics
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"模型已保存到 {save_path}")


def load_model(checkpoint_path, device, config=None):
    """加载模型检查点"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    loaded_config = checkpoint.get('config', config)
    
    if config is None:
        config = loaded_config
    
    # 创建集成模型
    model = IntegratedLLMPS(
        llm_config=config.get('llm_config'),
        temporal_encoder_config=config.get('temporal_encoder_config'),
        semantic_encoder_config=config.get('semantic_encoder_config'),
        fusion_config=config.get('fusion_config'),
        optimizer_config=config.get('optimizer_config'),
        forecasting_config=config.get('forecasting_config'),
        use_external_llm=config.get('use_external_llm', False),
        llm_checkpoint_path=config.get('llm_checkpoint_path'),
        use_attention=config.get('use_attention', True)
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, config


def train(args):
    """训练集成模型"""
    # 加载配置
    config = get_default_config()
    
    # 更新配置（如果提供了配置文件）
    if args.config_path:
        with open(args.config_path, 'r') as f:
            custom_config = json.load(f)
            # 递归更新配置
            def update_config(d, u):
                for k, v in u.items():
                    if isinstance(v, dict) and k in d:
                        d[k] = update_config(d[k], v)
                    else:
                        d[k] = v
                return d
            config = update_config(config, custom_config)
    
    # 添加LLM配置
    if 'llm_config' not in config:
        config['llm_config'] = {
            'vocab_size': 1024,
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 4,
            'dim_feedforward': 512,
            'dropout': 0.1,
            'max_seq_len': 128
        }
    
    # 添加集成配置
    config['use_external_llm'] = args.use_external_llm
    config['llm_checkpoint_path'] = args.llm_checkpoint_path
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"integrated_llm_ps_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.info(f"开始训练集成 LLM-PS 模型，输出目录: {output_dir}")
    logger.info(f"配置: {json.dumps(config, indent=2)}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建数据集
    logger.info(f"加载数据集，数据目录: {args.data_dir}")
    train_dataset = TimeSeriesDataset(
        root_dir=args.data_dir,
        split='train',
        seq_len=config['data']['seq_len'],
        forecast_len=config['data']['forecast_len'],
        stride=config['data']['train_stride'],
        transform=True
    )
    
    val_dataset = TimeSeriesDataset(
        root_dir=args.data_dir,
        split='val',
        seq_len=config['data']['seq_len'],
        forecast_len=config['data']['forecast_len'],
        stride=config['data']['val_stride'],
        transform=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 创建模型
    if args.checkpoint_path:
        logger.info(f"从检查点加载模型: {args.checkpoint_path}")
        model, loaded_config = load_model(args.checkpoint_path, device, config)
        # 更新配置
        config.update(loaded_config)
        start_epoch = loaded_config.get('epoch', 0) + 1
    else:
        logger.info("创建新的集成模型")
        model = IntegratedLLMPS(
            llm_config=config.get('llm_config'),
            temporal_encoder_config=config.get('temporal_encoder_config'),
            semantic_encoder_config=config.get('semantic_encoder_config'),
            fusion_config=config.get('fusion_config'),
            optimizer_config=config.get('optimizer_config'),
            forecasting_config=config.get('forecasting_config'),
            use_external_llm=config.get('use_external_llm', False),
            llm_checkpoint_path=config.get('llm_checkpoint_path'),
            use_attention=config.get('use_attention', True)
        ).to(device)
        start_epoch = 0
    
    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 如果从检查点加载，也加载优化器状态
    if args.checkpoint_path and 'optimizer_state_dict' in torch.load(args.checkpoint_path, map_location=device):
        optimizer.load_state_dict(torch.load(args.checkpoint_path, map_location=device)['optimizer_state_dict'])
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['training']['lr_factor'],
        patience=config['training']['lr_patience'],
        verbose=True
    )
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
    
    # 创建损失函数
    forecast_criterion = nn.MSELoss()
    
    # 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config['training']['epochs']):
        model.train()
        train_losses = []
        forecast_losses = []
        feature_losses = []
        prompt_losses = []
        llm_losses = []
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # 提取数据
            x_enc, x_mark_enc, y_true, x_dec, x_mark_dec = [b.to(device) for b in batch]
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(x_enc, x_mark_enc, mode='train')
            
            # 计算时间序列预测损失 (L_TIME)
            forecast = outputs['forecast']
            forecast_loss = forecast_criterion(forecast, y_true)
            
            # 计算特征学习损失 (L_FEAT)
            feature_loss = 0.0
            if 'semantic_features' in outputs and 'temporal_features' in outputs:
                # 计算语义特征和时间特征之间的一致性损失
                semantic_features = outputs['semantic_features']
                temporal_features = outputs['temporal_features']
                
                # 归一化特征向量，计算余弦相似度损失
                semantic_norm = F.normalize(semantic_features, p=2, dim=1)
                temporal_norm = F.normalize(temporal_features, p=2, dim=1)
                feature_loss = 1.0 - F.cosine_similarity(semantic_norm, temporal_norm, dim=1).mean()
            
            # 计算提示词生成损失
            prompt_loss = 0.0
            if 'prompt_logits' in outputs:
                prompt_logits = outputs['prompt_logits']
                # 简化处理，使用零标签
                prompt_target = torch.zeros(prompt_logits.size(0), dtype=torch.long, device=device)
                prompt_loss = F.cross_entropy(prompt_logits, prompt_target)
            
            # 计算LLM损失（如果有LLM输出）
            llm_loss = 0.0
            if 'llm_outputs' in outputs and outputs['llm_outputs'] is not None:
                # 在实际应用中，这里应该有真实的语言建模目标
                # 这里简化处理
                llm_outputs = outputs['llm_outputs']
                if 'logits' in llm_outputs:
                    logits = llm_outputs['logits']
                    # 简化：预测下一个token
                    llm_target = torch.zeros(logits.size(0), logits.size(1), dtype=torch.long, device=device)
                    llm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), llm_target.view(-1))
            
            # 总损失 L_OBJ = L_TIME + λL_FEAT + μL_PROMPT + γL_LLM
            feature_weight = config['training'].get('feature_weight', 0.5)  # λ
            prompt_weight = config['training'].get('prompt_weight', 0.1)    # μ
            llm_weight = config['training'].get('llm_weight', 0.1)          # γ
            
            loss = forecast_loss + feature_weight * feature_loss + prompt_weight * prompt_loss + llm_weight * llm_loss
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            
            # 更新参数
            optimizer.step()
            
            # 记录各个损失组件
            train_losses.append(loss.item())
            forecast_losses.append(forecast_loss.item())
            feature_losses.append(feature_loss.item() if isinstance(feature_loss, torch.Tensor) else feature_loss)
            prompt_losses.append(prompt_loss.item() if isinstance(prompt_loss, torch.Tensor) else prompt_loss)
            llm_losses.append(llm_loss.item() if isinstance(llm_loss, torch.Tensor) else llm_loss)
            
            # 打印进度
            if (batch_idx + 1) % config['training']['log_interval'] == 0:
                logger.info(f"Epoch [{epoch+1}/{config['training']['epochs']}] Batch [{batch_idx+1}/{len(train_loader)}] "
                           f"Loss: {loss.item():.4f} Forecast: {forecast_loss.item():.4f} "
                           f"Feature: {feature_loss.item() if isinstance(feature_loss, torch.Tensor) else feature_loss:.4f} "
                           f"Prompt: {prompt_loss.item() if isinstance(prompt_loss, torch.Tensor) else prompt_loss:.4f} "
                           f"LLM: {llm_loss.item() if isinstance(llm_loss, torch.Tensor) else llm_loss:.4f}")
        
        # 计算平均损失
        avg_train_loss = np.mean(train_losses)
        avg_forecast_loss = np.mean(forecast_losses)
        avg_feature_loss = np.mean(feature_losses)
        avg_prompt_loss = np.mean(prompt_losses)
        avg_llm_loss = np.mean(llm_losses)
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/train_forecast', avg_forecast_loss, epoch)
        writer.add_scalar('Loss/train_feature', avg_feature_loss, epoch)
        writer.add_scalar('Loss/train_prompt', avg_prompt_loss, epoch)
        writer.add_scalar('Loss/train_llm', avg_llm_loss, epoch)
        
        # 验证
        model.eval()
        val_losses = []
        val_forecast_losses = []
        val_metrics = {'mse': [], 'mae': [], 'rmse': [], 'mape': []}
        
        with torch.no_grad():
            for batch in val_loader:
                # 提取数据
                x_enc, x_mark_enc, y_true, x_dec, x_mark_dec = [b.to(device) for b in batch]
                
                # 前向传播，只评估预测性能
                forecast = model(x_enc, x_mark_enc, mode='forecast')
                
                # 计算预测损失
                forecast_loss = forecast_criterion(forecast, y_true)
                val_forecast_losses.append(forecast_loss.item())
                
                # 计算评估指标
                metrics = compute_metrics(forecast.cpu().numpy(), y_true.cpu().numpy())
                for k, v in metrics.items():
                    if k in val_metrics:
                        val_metrics[k].append(v)
        
        # 计算平均验证损失和指标
        avg_val_forecast_loss = np.mean(val_forecast_losses)
        avg_val_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/val_forecast', avg_val_forecast_loss, epoch)
        for k, v in avg_val_metrics.items():
            writer.add_scalar(f'Metrics/{k}', v, epoch)
        
        # 更新学习率
        scheduler.step(avg_val_forecast_loss)
        
        # 打印验证结果
        epoch_time = time.time() - start_time
        logger.info(f"Epoch [{epoch+1}/{config['training']['epochs']}] completed in {epoch_time:.2f}s")
        logger.info(f"Train Loss: {avg_train_loss:.4f}, Val Forecast Loss: {avg_val_forecast_loss:.4f}")
        logger.info(f"Val Metrics - MSE: {avg_val_metrics['mse']:.4f}, MAE: {avg_val_metrics['mae']:.4f}, "
                   f"RMSE: {avg_val_metrics['rmse']:.4f}, MAPE: {avg_val_metrics['mape']:.4f}")
        
        # 保存最佳模型
        if avg_val_forecast_loss < best_val_loss:
            best_val_loss = avg_val_forecast_loss
            save_path = os.path.join(output_dir, 'best_model.pth')
            save_model(model, optimizer, epoch, config, avg_val_metrics, save_path)
            logger.info(f"保存最佳模型，验证损失: {best_val_loss:.4f}")
        
        # 定期保存检查点
        if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
            save_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            save_model(model, optimizer, epoch, config, avg_val_metrics, save_path)
    
    # 保存最终模型
    save_path = os.path.join(output_dir, 'final_model.pth')
    save_model(model, optimizer, config['training']['epochs'] - 1, config, avg_val_metrics, save_path)
    
    # 关闭TensorBoard写入器
    writer.close()
    
    logger.info("训练完成!")
    return output_dir


def inference(args):
    """集成模型推理"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, 'integrated_inference')
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"开始集成推理，使用检查点: {args.checkpoint_path}")
    
    # 加载模型
    model, config = load_model(args.checkpoint_path, device)
    model.eval()
    
    # 加载输入数据（这里简化为随机数据）
    # 在实际应用中，应从指定路径加载数据
    if args.input_path:
        logger.info(f"从 {args.input_path} 加载输入数据")
        try:
            # 尝试加载数据
            input_data = np.load(args.input_path)
            input_data = torch.FloatTensor(input_data).to(device)
        except Exception as e:
            logger.warning(f"加载数据时出错: {e}，使用随机数据代替")
            # 使用随机数据
            input_data = torch.randn(1, config['temporal_encoder_config']['in_channels'], 
                                   config['data']['seq_len']).to(device)
    else:
        logger.info("未提供输入路径，使用随机数据")
        input_data = torch.randn(1, config['temporal_encoder_config']['in_channels'], 
                               config['data']['seq_len']).to(device)
    
    # 生成时间特征（简化）
    time_features = torch.zeros(1, config['data']['seq_len'], 8).to(device)
    
    # 推理
    with torch.no_grad():
        # 预测并生成解释
        forecast, explanation = model.generate_forecast_with_explanation(
            input_data, time_features, max_len=100, temperature=0.7
        )
    
    # 将预测结果转换为numpy数组
    forecast_np = forecast.cpu().numpy()
    
    # 保存结果
    forecast_path = os.path.join(output_dir, 'forecast.npy')
    np.save(forecast_path, forecast_np)
    logger.info(f"预测结果已保存到 {forecast_path}")
    
    # 如果有分词器，对生成的解释进行解码
    explanation_text = "解释token ID: " + str(explanation.cpu().numpy().tolist())
    with open(os.path.join(output_dir, 'explanation.txt'), 'w') as f:
        f.write(explanation_text)
    
    # 可视化结果
    if args.visualize:
        for j in range(min(3, forecast.shape[1])):  # 最多显示3个变量
            fig = plt.figure(figsize=(10, 6))
            plt.plot(forecast[0, j].cpu().numpy(), label='Forecast')
            plt.title(f"Forecast for Variable {j}")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'forecast_var_{j}.png'))
            plt.close(fig)
        logger.info(f"可视化已保存到 {output_dir}")
    
    return forecast


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='集成 LLM-PS 模型训练和推理')
    
    # 通用参数
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference'],
                        help='运行模式: train 或 inference')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录路径')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='模型检查点路径 (用于继续训练或推理)')
    parser.add_argument('--config_path', type=str, default=None,
                        help='配置文件路径 (JSON)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='禁用CUDA, 即使可用')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载的工作线程数')
    
    # LLM参数
    parser.add_argument('--use_external_llm', action='store_true',
                        help='是否使用外部LLM模型')
    parser.add_argument('--llm_checkpoint_path', type=str, default=None,
                        help='外部LLM模型检查点路径')
    
    # 推理参数
    parser.add_argument('--input_path', type=str, default=None,
                        help='用于推理的输入数据路径')
    parser.add_argument('--visualize', action='store_true',
                        help='在推理后生成可视化')
    
    args = parser.parse_args()
    
    # 根据模式执行不同的功能
    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        if not args.checkpoint_path:
            raise ValueError("推理模式需要提供--checkpoint_path")
        inference(args)


if __name__ == "__main__":
    main() 