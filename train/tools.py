import torch
import torch.nn as nn
from log.logger import get_logger

logger = get_logger(__name__, "tools.log")

class InstanceNorm3D(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, track_running_stats=True):
        """
        适用于金融时间序列的Instance Norm实现
        :param num_features: 特征维度（feature_dim）
        :param eps: 数值稳定性常数
        :param affine: 是否使用可学习仿射参数
        :param track_running_stats: 是否跟踪运行时统计量
        """
        super(InstanceNorm3D, self).__init__()
        self.inst_norm = nn.InstanceNorm1d(num_features, eps=eps, affine=affine, track_running_stats=track_running_stats)
    
    def forward(self, x):
        """
        输入数据格式: [batch_size, seq_len, feature_dim]
        输出数据格式: [batch_size, seq_len, feature_dim]
        """
        # 调整维度顺序以适应Instance Norm的输入要求 [N, C, L]
        x = x.permute(0, 2, 1)  # [batch_size, feature_dim, seq_len]
        
        # 执行Instance Norm
        x = self.inst_norm(x)
        
        # 恢复原始维度顺序
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, feature_dim]
        
        return x

# 示例用法
if __name__ == "__main__":
    # 假设输入数据为 [batch_size=32, seq_len=64, feature_dim=128]
    batch_size, seq_len, feature_dim = 16, 32, 64
    x = torch.randn(batch_size, seq_len, feature_dim)
    
    # 初始化Instance Norm层
    logger.info(f"初始化Instance Norm层之前: {x}")
    inst_norm = InstanceNorm3D(num_features=feature_dim)
    logger.info(f"初始化Instance Norm层之后: {inst_norm}")
    
    # 执行归一化
    normalized_x = inst_norm(x)
    print(f"归一化后数据形状: {normalized_x.shape}")  # 输出: torch.Size([32, 64, 128])