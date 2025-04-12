import logging
import os
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
torch.autograd.set_detect_anomaly(True)
from model.transformer import StockTransformerModel, StockPricePredictor
from data.data_processor import StockDataProcessor, StockDataset, create_dataloaders
from train.model_args import ModelArgs
from trainer import StockModelTrainer
from log.logger import get_logger

# 设置日志
logger = get_logger(__file__, log_file="train_model.log")


def train_model(args: ModelArgs):
    """
    训练股票预测模型的主函数
    
    参数:
        args: 命令行参数
    """
    logger.info("准备训练股票预测模型...")
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    logger.info(f"使用设备: {device}")
    
    # 设置随机种子以便结果可重现
    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logger.info(f"设置随机种子: {args.seed}")
    
    # 加载和处理数据
    logger.info("加载和处理数据...")

    # 创建数据处理器
    processor = StockDataProcessor(
        raw_data_dir=args.raw_data_dir,
        processed_data_dir=args.processed_data_dir,
        scaler_type=args.scaler_type
    )
    
    # 处理所有指定的股票
    tickers = args.tickers.split(',')
    data_results = {}
    
    for ticker in tickers:
        logger.info(f"处理股票 {ticker}...")
        
        # 如果已经处理过且不需要重新处理，直接加载
        if args.load_processed and os.path.exists(os.path.join(args.processed_data_dir, "train")):
            logger.info(f"加载已处理的数据 {ticker}...")
            
            # 加载序列数据
            train_seq = np.load(os.path.join(args.processed_data_dir, "train", "{}_train_sequences.npz".format(ticker)))
            val_seq = np.load(os.path.join(args.processed_data_dir, "eval", "{}_val_sequences.npz".format(ticker)))
            test_seq = np.load(os.path.join(args.processed_data_dir, "test", "{}_test_sequences.npz".format(ticker)))
            
            # 输出X, y 维度信息
            logger.info(f"训练集{ticker} :X, y 维度信息: {train_seq['X'].shape}, {train_seq['y'].shape}")
            
            # TODO: 需要优化，后续只需要加载训练数据，验证集和测试集数据在验证和测试中加载
            sequences = {
                'train': (train_seq['X'], train_seq['y']),
                'val': (val_seq['X'], val_seq['y']),
                'test': (test_seq['X'], test_seq['y'])
            }
            
            # 加载缩放器
            processor.load_scalers(ticker)
            
            data_results[ticker] = {
                'sequences': sequences
            }
            # 输出data_results维度信息
            logger.info(f"data_results维度信息: {data_results[ticker]['sequences']['train'][0].shape}, {data_results[ticker]['sequences']['train'][1].shape}")
        else:
            # 完整处理股票数据
            result = processor.process_stock_pipeline(
                ticker=ticker,
                source=args.data_source,
                test_size=args.test_size,
                val_size=args.val_size,
                sequence_length=args.sequence_length,
                prediction_horizon=args.prediction_horizon,
                feature_groups=args.feature_groups.split(',') if args.feature_groups else None,
                save_data=True
            )
            
            if result is None:
                logger.error(f"处理股票 {ticker} 失败，跳过...")
                continue
                
            data_results[ticker] = result
    
    # 检查是否有成功处理的数据
    if not data_results:
        logger.error("没有成功处理的股票数据，退出...")
        return
    
    # 创建数据加载器
    logger.info("创建数据加载器...")

    
    # 如果合并所有股票数据
    if args.merge_stocks and len(data_results) > 1:
        logger.info("合并所有股票数据...")
        
        # 合并序列
        all_sequences = {
            'train': ([], []),
            'val': ([], []),
            'test': ([], [])
        }
        
        for ticker, result in data_results.items():
            for split in ['train', 'val', 'test']:
                all_sequences[split][0].append(result['sequences'][split][0])
                all_sequences[split][1].append(result['sequences'][split][1])
        
        # 合并数组
        merged_sequences = {}
        for split in ['train', 'val', 'test']:
            X = np.concatenate(all_sequences[split][0], axis=0)
            y = np.concatenate(all_sequences[split][1], axis=0)
            merged_sequences[split] = (X, y)
            
        # 创建合并数据的加载器
        dataloaders = create_dataloaders(
            merged_sequences, 
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # 获取特征维度
        feature_dim = merged_sequences['train'][0].shape[-1]
    else:
        # 只使用第一只股票的数据（如果没有指定合并）
        ticker = list(data_results.keys())[0]
        logger.info(f"使用股票 {ticker} 的数据...")
        
        # 创建数据加载器
        dataloaders = create_dataloaders(
            data_results[ticker]['sequences'],
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # 获取特征维度
        feature_dim = data_results[ticker]['sequences']['train'][0].shape[-1]
    
    logger.info(f"特征维度: {feature_dim}")
    logger.info(f"训练批次数: {len(dataloaders['train'])}")
    logger.info(f"验证批次数: {len(dataloaders['val'])}")
    logger.info(f"测试批次数: {len(dataloaders['test'])}")
    
    # 创建模型
    logger.info("创建模型...")
    
    # 检查是否需要从检查点恢复
    if args.resume_from:
        logger.info(f"从检查点 {args.resume_from} 恢复模型...")
        checkpoint = torch.load(args.resume_from, map_location=device)
        
        # 创建模型
        model = StockTransformerModel(
            vocab_size=checkpoint["vocab_size"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            prediction_type=args.prediction_type,
            max_seq_len=args.max_sequence_length
        )
        
        # 加载权重
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("模型权重已加载")
    else:
        # 创建新模型
        # 使用args.norm设置归一化类型
        logger.info(f"使用归一化类型: {args.norm}")
        model = StockTransformerModel(
            vocab_size=feature_dim,
            hidden_size=args.hidden_size,
            dtype=args.dtype,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            qk_nope_head_dim=args.qk_nope_head_dim,
            qk_rope_head_dim=args.qk_rope_head_dim,
            v_head_dim=args.v_head_dim,
            moe_intermediate_size=args.moe_intermediate_size,
            num_experts=args.num_experts,
            num_experts_per_token=args.num_experts_per_token,
            max_seq_len=args.max_sequence_length,
            attention_dropout=args.attention_dropout,
            hidden_dropout=args.hidden_dropout,
            q_lora_rank=args.q_lora_rank,
            kv_lora_rank=args.kv_lora_rank,
            attention_scale_factor=args.attention_scale_factor,
            attention_type=args.attention_type,
            max_batch_size=args.max_batch_size,
            rope_scaling_factor=args.rope_scaling_factor,
            rope_theta=args.rope_theta,
            prediction_type=args.prediction_type,
            norm=args.norm  # 添加归一化参数
        ).to(device)
        logger.info("创建了新模型")
    # 设置配置
    train_config = {
        "learning_rate": args.learning_rate, # 学习率, 推荐值: 8e-5
        "weight_decay": args.weight_decay, # 权重衰减, 推荐值: 0.02
        "clip_grad_norm": args.clip_grad_norm, # 梯度裁剪范数, 推荐值: 1.0
        "num_epochs": args.num_epochs, # 训练轮次, 推荐值: 16
        "patience": args.patience, # 早停耐心, 推荐值: 3
        "save_dir": args.save_dir, # 模型保存目录, 推荐值: "models"
        "model_name": args.model_name, # 模型名称, 推荐值: "SpaceExploreAI"
        "log_interval": args.log_interval, # 日志记录间隔, 推荐值: 5
        "mixed_precision": not args.disable_mixed_precision, # 混合精度训练, 推荐值: True

        # 学习率调度器配置
        "scheduler_type": "epoch",        # 学习率调度器类型, 推荐值: "epoch"
        "scheduler_factor": args.scheduler_factor,        # 学习率衰减因子, 推荐值: 0.5
        "scheduler_patience": args.scheduler_patience,        # 调度器耐心值, 推荐值: 2
        "scheduler_threshold": args.scheduler_threshold,    # 改进阈值, 推荐值: 1e-4
        "scheduler_cooldown": args.scheduler_cooldown,        # 冷却期, 推荐值: 0
        "scheduler_min_lr": args.scheduler_min_lr,       # 最小学习率, 推荐值: 1e-6
        "scheduler_eps": args.scheduler_eps,           # 精度, 推荐值: 1e-8
        "scheduler_verbose": args.scheduler_verbose,      # 是否输出日志, 推荐值: True
        
        # 动态学习率调整配置
        "use_dynamic_lr": args.use_dynamic_lr,        # 是否使用动态学习率调整, 推荐值: False
        "trend_window_size": args.trend_window_size,         # 趋势窗口大小, 推荐值: 3
        "lr_boost_factor": args.lr_boost_factor,         # 学习率临时提升因子, 推荐值: 2.0
        "stagnation_threshold": args.stagnation_threshold,   # 损失停滞检测阈值, 推荐值: 0.01
        
        # 周期性学习率调整
        "use_cyclic_lr": args.use_cyclic_lr,         # 是否使用周期性学习率, 推荐值: False
        "cyclic_lr_base_size": args.cyclic_lr_base_size,       # 周期基础大小（轮次）, 推荐值: 5
        "cyclic_lr_max_factor": args.cyclic_lr_max_factor,   # 周期最大学习率因子, 推荐值: 10.0
        
        # 批次级学习率调整
        "batch_lr_update": args.batch_lr_update,               # 是否在批次级别调整学习率, 推荐值: False
        "batch_lr_update_steps": args.batch_lr_update_steps,   # 每多少批次调整一次学习率, 推荐值: 100
        "batch_lr_gamma": args.batch_lr_gamma,              # 批次级学习率衰减因子, 推荐值: 0.995
        
        # LLM-PS相关配置
        "use_llmps": args.use_llmps,                  # 是否使用LLM-PS进行时间序列增强, 推荐值: True
        "llmps_lambda_weight": args.llmps_lambda_weight, # λ值，约束损失权重, 推荐值: 0.01
        
        # MSCNN配置
        "mscnn_base_channels": args.mscnn_base_channels,  # 基础通道数, 推荐值: 64
        "mscnn_ms_blocks": args.mscnn_ms_blocks,      # 多尺度块数量, 推荐值: 3
        "mscnn_output_dim": args.mscnn_output_dim,    # 输出维度, 推荐值: 512
        
        # T2T配置
        "t2t_patch_size": args.t2t_patch_size,        # 时间片段大小, 推荐值: 24
        "t2t_overlap": args.t2t_overlap,              # 时间片段重叠大小, 推荐值: 8
        "t2t_embed_dim": args.t2t_embed_dim,          # 嵌入维度, 推荐值: 96
        "t2t_num_encoder_layers": args.t2t_num_encoder_layers, # 编码器层数, 推荐值: 4
        "t2t_num_decoder_layers": args.t2t_num_decoder_layers, # 解码器层数, 推荐值: 1
        "t2t_nhead": args.t2t_nhead,                  # 注意力头数, 推荐值: 4
        "t2t_dim_feedforward": args.t2t_dim_feedforward, # 前馈网络维度, 推荐值: 384
        "t2t_dropout": args.t2t_dropout,              # 丢弃率, 推荐值: 0.1
        "t2t_mask_ratio": args.t2t_mask_ratio,        # 掩码比率, 推荐值: 0.75
        "t2t_output_dim": args.t2t_output_dim,        # 输出维度, 推荐值: 512
    }
        
    # 创建优化器
    logger.info(f"创建优化器 AdamW (lr={train_config['learning_rate']}, weight_decay={train_config['weight_decay']})")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"])
    
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        threshold=args.scheduler_threshold,
        cooldown=args.scheduler_cooldown,
        min_lr=args.scheduler_min_lr
    )

    # 根据配置决定使用哪种学习率调度器
    if train_config["use_cyclic_lr"]:
        logger.info("创建周期性学习率调度器 CyclicLR")
        # 计算周期步数
        steps_per_epoch = len(dataloaders['train'])
        step_size_up = train_config["cyclic_lr_base_size"] * steps_per_epoch // 2
        
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=train_config["learning_rate"] / 10,  # 基础学习率
            max_lr=train_config["learning_rate"] * train_config["cyclic_lr_max_factor"],  # 最大学习率
            step_size_up=step_size_up,
            mode='triangular2',
            cycle_momentum=False
        )
        train_config["scheduler_type"] = "batch"  # 批次级调度器
    else:
        logger.info("创建学习率调度器 ReduceLROnPlateau")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=train_config["scheduler_factor"],
            patience=train_config["scheduler_patience"],
            threshold=train_config["scheduler_threshold"],
            threshold_mode='rel',
            cooldown=train_config["scheduler_cooldown"],
            min_lr=train_config["scheduler_min_lr"],
            eps=train_config["scheduler_eps"]
        )
        train_config["scheduler_type"] = "epoch"  # 轮次级调度器

            
        # 验证学习率调度器是否正确设置
        if not isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler) and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            raise TypeError(f"学习率调度器类型错误: {type(scheduler)}")
    
    # 训练配置
    # 创建训练器
    trainer = StockModelTrainer(
        model=model,
        train_dataloader=dataloaders['train'],
        val_dataloader=dataloaders['val'],
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=train_config,
        loss_fn_str=args.loss_fn_str
    )
    
    # 如果恢复训练，加载训练状态
    if args.resume_from:
        trainer.load_model(args.resume_from)
    
    # 训练模型
    logger.info("开始训练模型...")
    training_result = trainer.train(args.num_epochs)
    
    # 在测试集上评估模型
    logger.info("在测试集上评估模型...")
    test_loss, test_metrics = trainer.validate()
    
    # 打印测试结果
    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in test_metrics.items()])
    logger.info(f"测试结果: 损失: {test_loss:.4f}, 指标: {metrics_str}")
    
    # 保存最终模型
    logger.info("保存最终模型...")
    trainer.save_model(f"{args.model_name}_final.pt")
    
    logger.info("训练完成！")
    
    return trainer, test_metrics


if __name__ == "__main__":
    # 初始化模型参数
    args = ModelArgs()
    
    # 训练模型
    trainer, metrics = train_model(args) 