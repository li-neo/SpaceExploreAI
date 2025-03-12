import logging
import sys
import os
def init_logger(log_file="app.log"):
    # 创建以当前模块命名的Logger对象[3,5](@ref)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # 设置全局日志级别[6](@ref)

    # 定义日志格式[2,7](@ref)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 文件处理器：输出到当前目录指定文件[7](@ref)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_handler = logging.FileHandler(os.path.join(current_dir, log_file), encoding='utf-8')
    file_handler.setLevel(logging.INFO)  # 文件日志级别
    file_handler.setFormatter(formatter)

    # 控制台处理器（可选）[8](@ref)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)  # 控制台日志级别
    console_handler.setFormatter(formatter)

    # 添加处理器到Logger[4](@ref)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger