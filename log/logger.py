import logging
import sys
import os
from typing import Dict

# 存储所有已创建的logger
loggers: Dict[str, logging.Logger] = {}

# 获取当前文件所在目录的绝对路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 设置日志目录
LOG_DIR = os.path.join(CURRENT_DIR, "logs")

def get_logger(name: str = None, log_file: str = "app.log") -> logging.Logger:
    """
    获取或创建logger实例
    
    参数:
        name: logger名称，默认使用调用模块的名称
        log_file: 日志文件名
        
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
    
    # 禁止日志传播到父级logger，防止重复输出
    logger.propagate = False
    
    # 确保日志目录存在
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 构建完整的日志文件路径
    log_file_path = os.path.join(LOG_DIR, log_file)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file_path)
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