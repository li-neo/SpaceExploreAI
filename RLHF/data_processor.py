import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import logging
from log.logger import get_logger

# 设置日志
logger = get_logger(__file__, log_file="dpo_trainer.log")

class PreferenceDataset(Dataset):
    """
    偏好数据集，用于DPO训练
    
    数据格式为(输入序列, 胜者序列, 败者序列)的三元组
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_prompt_length: int = 512,
        max_response_length: int = 128,
        scaler_type: str = "robust"
    ):
        """
        初始化偏好数据集
        
        参数:
            data_path: 数据路径
            split: 数据集划分，'train'或'eval'
            max_prompt_length: 最大输入序列长度
            max_response_length: 最大响应序列长度
            scaler_type: 标准化类型
        """
        self.data_path = data_path
        self.split = split
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.scaler_type = scaler_type
        
        # 加载数据
        self.data = self._load_data()
        logger.info(f"加载 {split} 数据集，共 {len(self.data)} 条样本")
    
    def _load_data(self) -> List[Dict]:
        """
        加载偏好数据
        
        返回:
            数据列表
        """
        # 检查路径是否存在
        split_path = os.path.join(self.data_path, f"{self.split}.jsonl")
        if not os.path.exists(split_path):
            # 如果路径不存在，可以尝试从CSV文件读取
            logger.warning(f"JSONL文件不存在: {split_path}，尝试从CSV文件读取...")
            csv_path = os.path.join(self.data_path, f"{self.split}.csv")
            if os.path.exists(csv_path):
                return self._load_from_csv(csv_path)
            else:
                logger.error(f"数据文件不存在: {split_path} 或 {csv_path}")
                return []
        
        # 从JSONL文件读取
        data = []
        with open(split_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError:
                    logger.warning(f"无法解析JSONL行: {line}")
        
        return data
    
    def _load_from_csv(self, file_path: str) -> List[Dict]:
        """
        从CSV文件加载偏好数据
        
        参数:
            file_path: CSV文件路径
            
        返回:
            数据列表
        """
        try:
            df = pd.read_csv(file_path)
            data = []
            for _, row in df.iterrows():
                item = {
                    "prompt": row.get("prompt"),
                    "chosen": row.get("chosen"),
                    "rejected": row.get("rejected")
                }
                # 确保所有字段都有值
                if all(item.values()):
                    data.append(item)
                else:
                    logger.warning(f"跳过不完整的行: {row}")
            return data
        except Exception as e:
            logger.error(f"从CSV加载数据出错: {e}")
            return []
    
    def __len__(self) -> int:
        """返回数据集长度"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取数据项
        
        参数:
            idx: 索引
            
        返回:
            包含输入、胜者和败者序列的字典
        """
        item = self.data[idx]
        
        # 从数据中获取三元组
        prompt = item.get("prompt")
        chosen = item.get("chosen")
        rejected = item.get("rejected")
        
        # 返回格式化的数据
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        }


class PreferenceDataProcessor:
    """
    偏好数据处理器，用于处理和准备DPO训练数据
    """
    
    def __init__(
        self,
        data_path: str,
        max_prompt_length: int = 512,
        max_response_length: int = 128,
        scaler_type: str = "robust",
        eval_ratio: float = 0.1
    ):
        """
        初始化数据处理器
        
        参数:
            data_path: 数据路径
            max_prompt_length: 最大输入序列长度
            max_response_length: 最大响应序列长度
            scaler_type: 标准化类型
            eval_ratio: 评估集比例
        """
        self.data_path = data_path
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.scaler_type = scaler_type
        self.eval_ratio = eval_ratio
        
        # 确保数据目录存在
        os.makedirs(data_path, exist_ok=True)
    
    def process_data(self, input_data: List[Dict]) -> None:
        """
        处理输入数据并保存为训练和评估数据集
        
        参数:
            input_data: 原始偏好数据
        """
        # 打乱数据
        np.random.shuffle(input_data)
        
        # 划分训练集和评估集
        eval_size = int(len(input_data) * self.eval_ratio)
        train_data = input_data[eval_size:]
        eval_data = input_data[:eval_size]
        
        # 保存数据
        self._save_data(train_data, "train")
        self._save_data(eval_data, "eval")
        
        logger.info(f"数据处理完成: 训练集 {len(train_data)} 条，评估集 {len(eval_data)} 条")
    
    def _save_data(self, data: List[Dict], split: str) -> None:
        """
        保存数据到JSONL文件
        
        参数:
            data: 数据列表
            split: 数据集划分名称
        """
        output_file = os.path.join(self.data_path, f"{split}.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        logger.info(f"保存 {split} 数据到 {output_file}")
    
    def create_dataloaders(self, batch_size: int) -> Dict[str, DataLoader]:
        """
        创建训练和评估数据加载器
        
        参数:
            batch_size: 批次大小
            
        返回:
            包含训练和评估数据加载器的字典
        """
        # 创建数据集
        train_dataset = PreferenceDataset(
            self.data_path,
            split="train",
            max_prompt_length=self.max_prompt_length,
            max_response_length=self.max_response_length,
            scaler_type=self.scaler_type
        )
        
        eval_dataset = PreferenceDataset(
            self.data_path,
            split="eval",
            max_prompt_length=self.max_prompt_length,
            max_response_length=self.max_response_length,
            scaler_type=self.scaler_type
        )
        
        # 创建数据加载器
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        return {
            "train": train_dataloader,
            "eval": eval_dataloader
        }


def create_synthetic_preference_data(
    model,
    num_samples: int = 1000,
    sequence_length: int = 32,
    prediction_horizon: int = 2,
    variation_factor: float = 0.1,
    output_path: str = "data/preferences"
) -> None:
    """
    创建合成偏好数据
    
    参数:
        model: 预训练模型
        num_samples: 样本数
        sequence_length: 序列长度
        prediction_horizon: 预测周期
        variation_factor: 变异因子
        output_path: 输出路径
    """
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 将模型设置为评估模式
    model.eval()
    device = next(model.parameters()).device
    
    preference_data = []
    
    # 生成随机输入序列
    for i in range(num_samples):
        # 创建随机输入序列
        input_sequence = torch.randn(1, sequence_length, model.vocab_size).to(device)
        
        # 获取模型预测
        with torch.no_grad():
            output = model(input_sequence)
            
        # 获取预测序列
        prediction = output.detach().cpu().numpy()
        
        # 生成更好和更差的预测
        # 更好的预测：将原始预测稍微向真实值偏移
        better_prediction = prediction * (1.0 + variation_factor * np.random.randn(*prediction.shape))
        
        # 更差的预测：将原始预测向远离真实值的方向偏移
        worse_prediction = prediction * (1.0 + variation_factor * 2 * np.random.randn(*prediction.shape))
        
        # 将数据添加到偏好数据列表
        preference_data.append({
            "prompt": input_sequence.detach().cpu().numpy().tolist(),
            "chosen": better_prediction.tolist(),
            "rejected": worse_prediction.tolist()
        })
    
    # 处理和保存数据
    processor = PreferenceDataProcessor(
        data_path=output_path,
        max_prompt_length=sequence_length,
        max_response_length=prediction_horizon
    )
    
    processor.process_data(preference_data)
    logger.info(f"创建了 {num_samples} 条合成偏好数据并保存到 {output_path}") 