"""
时间序列数据集处理模块
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from log.logger import get_logger

# 创建logger
logger = get_logger("dataset", "llmps_model.log")

class TimeSeriesDataset(Dataset):
    """
    时间序列数据集
    
    参数:
        root_dir (str): 数据根目录
        split (str): 数据集划分，'train', 'val' 或 'test'
        seq_len (int): 输入序列长度
        forecast_len (int): 预测长度
        stride (int): 序列采样步长
        transform (bool): 是否应用数据变换
    """
    def __init__(self, root_dir, split='train', seq_len=96, forecast_len=96, stride=24, transform=False):
        self.root_dir = root_dir
        self.split = split
        self.seq_len = seq_len
        self.forecast_len = forecast_len
        self.stride = stride
        self.transform = transform
        
        # 加载数据
        self.data = self._load_data()
        
        # 生成样本索引
        self.indices = self._generate_indices()
        
        logger.info(f"创建 {split} 数据集，样本数: {len(self.indices)}")
    
    def _load_data(self):
        """加载数据"""
        # 在实际应用中，应从文件加载数据
        # 这里简化为生成随机数据
        logger.warning("使用随机生成的数据，在实际应用中应替换为真实数据")
        
        # 生成随机数据：[total_len, num_variables]
        total_len = 1000 if self.split == 'train' else 200
        num_variables = 1
        
        # 确定数据文件路径
        try:
            # 尝试加载真实数据
            data_path = os.path.join(self.root_dir, f"{self.split}.npy")
            if os.path.exists(data_path):
                data = np.load(data_path)
                logger.info(f"从 {data_path} 加载数据")
                return data
        except Exception as e:
            logger.warning(f"无法加载数据: {e}")
        
        # 生成随机数据
        data = np.random.randn(total_len, num_variables)
        
        return data
    
    def _generate_indices(self):
        """生成样本索引"""
        data_len = len(self.data)
        indices = []
        
        # 遍历数据，按步长生成索引
        for i in range(0, data_len - self.seq_len - self.forecast_len + 1, self.stride):
            indices.append(i)
        
        return indices
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.indices)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        # 获取起始索引
        start_idx = self.indices[idx]
        
        # 获取编码器输入序列
        x_enc = self.data[start_idx:start_idx + self.seq_len].copy()
        
        # 获取目标序列
        y_true = self.data[start_idx + self.seq_len:start_idx + self.seq_len + self.forecast_len].copy()
        
        # 获取解码器输入序列（简化，与编码器输入相同）
        x_dec = x_enc.copy()
        
        # 创建时间特征（简化，使用位置编码）
        x_mark_enc = np.arange(self.seq_len).reshape(-1, 1) / self.seq_len
        x_mark_dec = np.arange(self.seq_len).reshape(-1, 1) / self.seq_len
        
        # 应用数据变换
        if self.transform:
            # 简单缩放
            x_enc = (x_enc - np.mean(x_enc)) / (np.std(x_enc) + 1e-10)
            x_dec = (x_dec - np.mean(x_dec)) / (np.std(x_dec) + 1e-10)
        
        # 转换为torch张量并调整维度
        # 从 [seq_len, num_variables] 到 [num_variables, seq_len]
        x_enc = torch.FloatTensor(x_enc.transpose())
        y_true = torch.FloatTensor(y_true.transpose())
        x_dec = torch.FloatTensor(x_dec.transpose())
        x_mark_enc = torch.FloatTensor(x_mark_enc)
        x_mark_dec = torch.FloatTensor(x_mark_dec)
        
        return x_enc, x_mark_enc, y_true, x_dec, x_mark_dec 