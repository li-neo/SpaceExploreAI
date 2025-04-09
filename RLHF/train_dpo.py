import os
import argparse
import torch
import logging
import sys
import random
import numpy as np
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from log.logger import get_logger
from RLHF.dpo_args import DPOArgs
from RLHF.dpo_trainer import DPOTrainer
from RLHF.data_processor import PreferenceDataProcessor, create_synthetic_preference_data
from model.transformer import StockTransformerModel
from llmps_model.models.llm_ps import LLMPS

# 设置日志
logger = get_logger(__file__, log_file="dpo_training.log")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="使用DPO训练SpaceExploreAI模型")
    
    # 基本参数
    parser.add_argument("--beta", type=float, default=0.1, help="DPO的KL散度正则化系数")
    parser.add_argument("--reference_model_path", type=str, required=True, help="参考模型路径")
    parser.add_argument("--preference_data_path", type=str, default="data/preferences", help="偏好数据路径")
    parser.add_argument("--create_synthetic_data", action="store_true", help="是否创建合成偏好数据")
    parser.add_argument("--num_synthetic_samples", type=int, default=1000, help="合成样本数量")
    
    # 训练参数
    parser.add_argument("--dpo_batch_size", type=int, default=4, help="DPO训练批次大小")
    parser.add_argument("--dpo_learning_rate", type=float, default=5e-6, help="DPO训练学习率")
    parser.add_argument("--dpo_num_epochs", type=int, default=3, help="DPO训练轮数")
    parser.add_argument("--eval_every", type=int, default=100, help="每多少步评估一次")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--save_dir", type=str, default="models/dpo", help="模型保存目录")
    parser.add_argument("--model_name", type=str, default="SpaceExploreAI_DPO", help="模型名称")
    
    # 设备参数
    parser.add_argument("--device", type=str, default=None, help="训练设备")
    
    args = parser.parse_args()
    return args


def set_seed(seed):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"设置随机种子: {seed}")


def load_model(model_path, device=None):
    """
    加载预训练模型
    
    参数:
        model_path: 模型路径
        device: 设备
        
    返回:
        加载的模型
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    # 检查模型路径存在
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return None
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    
    # 检查是否是LLMPS模型
    if 'llmps' in model_path.lower():
        # 创建LLM-PS模型
        logger.info("加载LLM-PS模型...")
        
        # 从检查点获取配置
        temporal_encoder_config = checkpoint.get("temporal_encoder_config", {})
        semantic_encoder_config = checkpoint.get("semantic_encoder_config", {})
        fusion_config = checkpoint.get("fusion_config", {})
        optimizer_config = checkpoint.get("optimizer_config", {})
        forecasting_config = checkpoint.get("forecasting_config", {})
        
        # 创建模型
        model = LLMPS(
            temporal_encoder_config=temporal_encoder_config,
            semantic_encoder_config=semantic_encoder_config,
            fusion_config=fusion_config,
            optimizer_config=optimizer_config,
            forecasting_config=forecasting_config
        )
    else:
        # 创建普通Transformer模型
        logger.info("加载Transformer模型...")
        
        # 从检查点获取必要参数
        vocab_size = checkpoint.get("vocab_size", 64)
        hidden_size = checkpoint.get("hidden_size", 256)
        
        # 创建模型
        model = StockTransformerModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=checkpoint.get("num_layers", 4),
            num_heads=checkpoint.get("num_heads", 4),
            max_seq_len=checkpoint.get("max_seq_len", 128),
            prediction_type=checkpoint.get("prediction_type", "regression")
        )
    
    # 加载模型权重
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"成功从 {model_path} 加载模型")
    
    return model.to(device)


def setup_dpo_training():
    """设置并启动DPO训练"""
    # 解析命令行参数
    cmd_args = parse_args()
    
    # 设置随机种子
    set_seed(cmd_args.seed)
    
    # 设置设备
    if cmd_args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(cmd_args.device)
    logger.info(f"使用设备: {device}")
    
    # 创建DPO参数
    dpo_args = DPOArgs()
    
    # 更新参数
    dpo_args.beta = cmd_args.beta
    dpo_args.reference_model_path = cmd_args.reference_model_path
    dpo_args.preference_data_path = cmd_args.preference_data_path
    dpo_args.dpo_batch_size = cmd_args.dpo_batch_size
    dpo_args.dpo_learning_rate = cmd_args.dpo_learning_rate
    dpo_args.dpo_num_epochs = cmd_args.dpo_num_epochs
    dpo_args.eval_every = cmd_args.eval_every
    dpo_args.save_dir = cmd_args.save_dir
    dpo_args.model_name = cmd_args.model_name
    dpo_args.seed = cmd_args.seed
    
    # 确保保存目录存在
    os.makedirs(dpo_args.save_dir, exist_ok=True)
    
    # 加载参考模型（通常是在监督训练后的模型）
    logger.info(f"加载参考模型: {dpo_args.reference_model_path}")
    reference_model = load_model(dpo_args.reference_model_path, device)
    
    # 创建要优化的模型副本
    logger.info("创建要优化的模型副本")
    model = load_model(dpo_args.reference_model_path, device)
    
    # 数据处理
    # 如果需要，创建合成偏好数据
    if cmd_args.create_synthetic_data:
        logger.info(f"创建 {cmd_args.num_synthetic_samples} 条合成偏好数据")
        create_synthetic_preference_data(
            model=reference_model,
            num_samples=cmd_args.num_synthetic_samples,
            sequence_length=dpo_args.sequence_length,
            prediction_horizon=dpo_args.prediction_horizon,
            output_path=dpo_args.preference_data_path
        )
    
    # 创建数据处理器和数据加载器
    logger.info("创建数据加载器")
    data_processor = PreferenceDataProcessor(
        data_path=dpo_args.preference_data_path,
        max_prompt_length=dpo_args.max_prompt_length,
        max_response_length=dpo_args.max_response_length,
        eval_ratio=dpo_args.eval_ratio
    )
    
    dataloaders = data_processor.create_dataloaders(batch_size=dpo_args.dpo_batch_size)
    
    # 创建DPO训练器
    logger.info("创建DPO训练器")
    trainer = DPOTrainer(
        model=model,
        reference_model=reference_model,
        train_dataloader=dataloaders["train"],
        eval_dataloader=dataloaders["eval"],
        args=dpo_args,
        device=device
    )
    
    # 开始训练
    logger.info("开始DPO训练")
    history = trainer.train()
    
    # 绘制训练曲线
    trainer.plot_training_curves()
    
    # 返回训练后的模型
    return trainer.model


if __name__ == "__main__":
    try:
        setup_dpo_training()
        logger.info("DPO训练完成！")
    except Exception as e:
        logger.error(f"DPO训练出错: {e}", exc_info=True)
        raise 