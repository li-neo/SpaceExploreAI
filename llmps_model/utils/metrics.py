"""
评估指标计算模块
"""

import numpy as np
from log.logger import get_logger

# 创建logger
logger = get_logger("metrics", "llmps_model.log")

def compute_metrics(pred, true):
    """
    计算时间序列预测的评估指标
    
    参数:
        pred (numpy.ndarray): 预测值
        true (numpy.ndarray): 真实值
        
    返回:
        dict: 包含各项评估指标的字典
    """
    metrics = {}
    
    # Mean Squared Error (MSE)
    metrics['mse'] = np.mean(np.square(pred - true))
    
    # Mean Absolute Error (MAE)
    metrics['mae'] = np.mean(np.abs(pred - true))
    
    # Root Mean Squared Error (RMSE)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # Mean Absolute Percentage Error (MAPE)
    # 避免除以0
    epsilon = 1e-10
    abs_percentage_error = np.abs((pred - true) / (np.abs(true) + epsilon))
    metrics['mape'] = np.mean(abs_percentage_error) * 100
    
    logger.debug(f"计算指标 - MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, MAPE: {metrics['mape']:.4f}")
    
    return metrics 