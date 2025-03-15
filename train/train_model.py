import logging
import os
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from model.transformer import StockTransformerModel, StockPricePredictor
from data.data_processor import StockDataProcessor, StockDataset, create_dataloaders
from trainer import StockModelTrainer
from log.logger import get_logger

# 设置日志
logger = get_logger(__file__, log_file="train_model.log")


def train_stock_model(args):
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
        if args.load_processed and os.path.exists(os.path.join(args.processed_data_dir, ticker)):
            logger.info(f"加载已处理的数据 {ticker}...")
            
            # 加载序列数据
            train_seq = np.load(os.path.join(args.processed_data_dir, ticker, "train_sequences.npz"))
            val_seq = np.load(os.path.join(args.processed_data_dir, ticker, "val_sequences.npz"))
            test_seq = np.load(os.path.join(args.processed_data_dir, ticker, "test_sequences.npz"))
            
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
            max_seq_len=args.sequence_length
        )
        
        # 加载权重
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("模型权重已加载")
    else:
        # 创建新模型
        model = StockTransformerModel(
            vocab_size=feature_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            qk_nope_head_dim=args.qk_nope_head_dim,
            qk_rope_head_dim=args.qk_rope_head_dim,
            v_head_dim=args.v_head_dim,
            moe_intermediate_size=args.moe_intermediate_size,
            num_experts=args.num_experts,
            num_experts_per_token=args.num_experts_per_token,
            max_seq_len=args.sequence_length,
            attention_dropout=args.attention_dropout,
            hidden_dropout=args.hidden_dropout,
            use_mixed_attention=not args.disable_mixed_attention,
            prediction_type=args.prediction_type
        )
        logger.info("创建了新模型")
        
    # 设置训练配置
    train_config = {
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "clip_grad_norm": args.clip_grad_norm,
        "num_epochs": args.num_epochs,
        "patience": args.patience,
        "save_dir": args.save_dir,
        "model_name": args.model_name,
        "log_interval": args.log_interval,
        "mixed_precision": not args.disable_mixed_precision
    }
    
    # 创建训练器
    trainer = StockModelTrainer(
        model=model,
        train_dataloader=dataloaders['train'],
        val_dataloader=dataloaders['val'],
        device=device,
        config=train_config
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
    parser = argparse.ArgumentParser(description="训练股票预测模型")
    
    # 数据相关参数
    parser.add_argument("--raw_data_dir", type=str, default="../data/raw", help="原始数据目录")
    parser.add_argument("--processed_data_dir", type=str, default="../data/processed", help="处理后数据目录")
    parser.add_argument("--tickers", type=str, default="AAPL", help="股票代码，多个用逗号分隔")
    parser.add_argument("--data_source", type=str, default="yahoo", help="数据源，如'yahoo'或'alphavantage'")
    parser.add_argument("--load_processed", action="store_true", help="加载已处理的数据")
    parser.add_argument("--merge_stocks", action="store_true", help="合并多只股票的数据")
    parser.add_argument("--scaler_type", type=str, default="robust", help="缩放器类型，如'standard'、'minmax'或'robust'")
    parser.add_argument("--test_size", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--val_size", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--sequence_length", type=int, default=60, help="序列长度")
    parser.add_argument("--prediction_horizon", type=int, default=5, help="预测周期")
    parser.add_argument("--feature_groups", type=str, default=None, help="特征组，多个用逗号分隔")
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    
    # 模型相关参数
    parser.add_argument("--hidden_size", type=int, default=256, help="隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=4, help="Transformer层数")
    parser.add_argument("--num_heads", type=int, default=4, help="注意力头数量")
    parser.add_argument("--qk_nope_head_dim", type=int, default=32, help="不使用旋转位置编码的Q/K头维度")
    parser.add_argument("--qk_rope_head_dim", type=int, default=32, help="使用旋转位置编码的Q/K头维度")
    parser.add_argument("--v_head_dim", type=int, default=64, help="值向量的头维度")
    parser.add_argument("--moe_intermediate_size", type=int, default=256, help="MoE中间层维度")
    parser.add_argument("--num_experts", type=int, default=8, help="专家数量")
    parser.add_argument("--num_experts_per_token", type=int, default=2, help="每个token使用的专家数量")
    parser.add_argument("--attention_dropout", type=float, default=0.1, help="注意力Dropout比率")
    parser.add_argument("--hidden_dropout", type=float, default=0.1, help="隐藏层Dropout比率")
    parser.add_argument("--disable_mixed_attention", action="store_true", help="禁用混合潜在注意力")
    parser.add_argument("--prediction_type", type=str, default="regression", help="预测类型，'regression'或'classification'")
    
    # 训练相关参数
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="梯度裁剪范数")
    parser.add_argument("--num_epochs", type=int, default=30, help="训练轮次")
    parser.add_argument("--patience", type=int, default=5, help="早停耐心")
    parser.add_argument("--save_dir", type=str, default="./models", help="模型保存目录")
    parser.add_argument("--model_name", type=str, default="stock_transformer", help="模型名称")
    parser.add_argument("--log_interval", type=int, default=10, help="日志记录间隔")
    parser.add_argument("--device", type=str, default="cuda", help="训练设备")
    parser.add_argument("--disable_mixed_precision", action="store_true", help="禁用混合精度训练")
    parser.add_argument("--resume_from", type=str, default=None, help="从检查点恢复训练")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 训练模型
    trainer, metrics = train_stock_model(args) 