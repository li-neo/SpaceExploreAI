#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM-PS主脚本
用于训练、评估和推理
"""
import argparse
import json
from datetime import datetime

from log.logger import get_logger

# 创建logger
logger = get_logger("main", "llmps_model.log")

# 导入相关模块
from llmps_model.models.llm_ps import LLMPS
from llmps_model.data.dataset import TimeSeriesDataset
from llmps_model.utils.metrics import compute_metrics
from llmps_model.utils.visualization import plot_predictions
from llmps_model.configs.default_config import get_default_config

# 这里引入集成训练模块的函数，以便复用
from llmps_model.train_integrated import train, inference, save_model, load_model

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LLM-PS模型训练、评估和推理')
    
    # 通用参数
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'eval', 'inference'],
                        help='运行模式: train, eval 或 inference')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录路径')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='模型检查点路径 (用于继续训练、评估或推理)')
    parser.add_argument('--config_path', type=str, default=None,
                        help='配置文件路径 (JSON)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='禁用CUDA, 即使可用')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载的工作线程数')
    
    # 推理参数
    parser.add_argument('--input_path', type=str, default=None,
                        help='用于推理的输入数据路径')
    parser.add_argument('--visualize', action='store_true',
                        help='在推理后生成可视化')
    
    args = parser.parse_args()
    
    logger.info(f"启动LLM-PS模型，模式: {args.mode}")
    
    # 根据模式执行不同的功能
    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        # 在评估模式下使用验证集进行评估
        args.eval_mode = True
        train(args)
    elif args.mode == 'inference':
        if not args.checkpoint_path:
            raise ValueError("推理模式需要提供--checkpoint_path")
        inference(args)
    
    logger.info("LLM-PS模型运行完成")


if __name__ == "__main__":
    main() 