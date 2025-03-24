"""
时间序列可视化模块
"""

import matplotlib.pyplot as plt
import numpy as np
from log.logger import get_logger

# 创建logger
logger = get_logger("visualization", "llmps_model.log")

def plot_predictions(true, pred, title="时间序列预测", save_path=None):
    """
    绘制时间序列预测结果
    
    参数:
        true (numpy.ndarray): 真实值，形状为 [batch_size, num_variables, seq_len]
        pred (numpy.ndarray): 预测值，形状为 [batch_size, num_variables, seq_len]
        title (str): 图表标题
        save_path (str, 可选): 图像保存路径，如果为None则不保存
        
    返回:
        matplotlib.figure.Figure: 图表对象
    """
    batch_size, num_variables, seq_len = true.shape
    
    fig, axes = plt.subplots(min(batch_size, 3), num_variables, figsize=(15, 10))
    
    # 确保axes是二维的
    if batch_size == 1 and num_variables == 1:
        axes = np.array([[axes]])
    elif batch_size == 1:
        axes = axes.reshape(1, -1)
    elif num_variables == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(min(batch_size, 3)):
        for j in range(num_variables):
            ax = axes[i, j]
            ax.plot(true[i, j], 'b-', label='True')
            ax.plot(pred[i, j], 'r--', label='Predicted')
            ax.set_title(f"Sample {i+1}, Variable {j+1}")
            ax.grid(True)
            ax.legend()
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path)
        logger.info(f"图像已保存到 {save_path}")
    
    return fig 