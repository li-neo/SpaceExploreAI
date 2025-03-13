import logging
import sys
import os
from typing import Dict

# 存储所有已创建的logger
loggers: Dict[str, logging.Logger] = {}

def get_logger(name: str = None, log_file: str = "app.log") -> logging.Logger:
    """
    获取或创建logger实例
    
    参数:
        name: logger名称，默认使用调用模块的名称
        log_file: 日志文件路径
        
    返回:
        配置好的logger实例
    """
    if name is None:
        name = "root"
    
    # 如果logger已经存在，直接返回
    if name in loggers:
        return loggers[name]
    
    # 创建新的logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 确保日志目录存在
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # 清除已存在的处理器（避免重复）
    logger.handlers.clear()
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 存储logger实例
    loggers[name] = logger
    
    return logger