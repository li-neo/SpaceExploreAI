#!/usr/bin/env python3
"""
股价预测模型快速入门示例

本脚本展示了使用SpaceExploreAI库训练和评估股票预测模型的完整流程：
1. 下载股票数据
2. 处理数据
3. 训练模型
4. 评估模型
"""

import os
import argparse
import logging
import sys
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(command):
    """执行命令并记录"""
    logger.info(f"执行命令: {command}")
    os.system(command)


def main(args):
    """主函数"""
    # 1. 下载股票数据
    logger.info("=== 步骤 1: 下载股票数据 ===")
    download_cmd = (
        f"python -m SpaceExploreAI.data.download_data "
        f"--tickers {args.tickers} "
        f"--start_date {args.start_date} "
        f"--end_date {args.end_date} "
        f"--output_dir {args.data_dir}/raw "
        f"--source {args.data_source}"
    )
    run_command(download_cmd)
    
    # 2. 训练模型
    logger.info("=== 步骤 2: 训练模型 ===")
    train_cmd = (
        f"python -m SpaceExploreAI.train.train_model "
        f"--tickers {args.tickers} "
        f"--raw_data_dir {args.data_dir}/raw "
        f"--processed_data_dir {args.data_dir}/processed "
        f"--sequence_length {args.sequence_length} "
        f"--prediction_horizon {args.prediction_horizon} "
        f"--batch_size {args.batch_size} "
        f"--hidden_size {args.hidden_size} "
        f"--num_layers {args.num_layers} "
        f"--num_heads {args.num_heads} "
        f"--num_epochs {args.num_epochs} "
        f"--prediction_type {args.prediction_type} "
        f"--save_dir {args.save_dir} "
        f"--model_name {args.model_name}"
    )
    run_command(train_cmd)
    
    # 3. 评估模型
    logger.info("=== 步骤 3: 评估模型 ===")
    eval_cmd = (
        f"python -m SpaceExploreAI.evaluate.evaluate_model "
        f"--model_path {args.save_dir}/{args.model_name}_best.pt "
        f"--test_data {args.tickers} "
        f"--processed_data_dir {args.data_dir}/processed "
        f"--output_dir {args.results_dir} "
        f"--batch_size {args.batch_size}"
    )
    run_command(eval_cmd)
    
    logger.info("=== 快速入门流程已完成! ===")
    logger.info(f"训练好的模型保存在: {args.save_dir}/{args.model_name}_best.pt")
    logger.info(f"评估结果保存在: {args.results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="股票预测模型快速入门")
    
    # 数据参数
    parser.add_argument("--tickers", type=str, default="AAPL", help="股票代码，多个用逗号分隔")
    parser.add_argument("--start_date", type=str, default="2010-01-01", help="开始日期")
    parser.add_argument("--end_date", type=str, default=datetime.now().strftime("%Y-%m-%d"), help="结束日期")
    parser.add_argument("--data_source", type=str, default="yahoo", help="数据源")
    parser.add_argument("--data_dir", type=str, default="./data", help="数据目录")
    
    # 模型参数
    parser.add_argument("--sequence_length", type=int, default=60, help="序列长度")
    parser.add_argument("--prediction_horizon", type=int, default=5, help="预测周期")
    parser.add_argument("--hidden_size", type=int, default=256, help="隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=4, help="Transformer层数")
    parser.add_argument("--num_heads", type=int, default=4, help="注意力头数量")
    parser.add_argument("--prediction_type", type=str, default="regression", help="预测类型，'regression'或'classification'")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮次")
    parser.add_argument("--save_dir", type=str, default="./models", help="模型保存目录")
    parser.add_argument("--model_name", type=str, default="stock_transformer", help="模型名称")
    parser.add_argument("--results_dir", type=str, default="./results", help="结果目录")
    
    args = parser.parse_args()
    
    # 创建必要的目录
    os.makedirs(f"{args.data_dir}/raw", exist_ok=True)
    os.makedirs(f"{args.data_dir}/processed", exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 运行主函数
    main(args) 