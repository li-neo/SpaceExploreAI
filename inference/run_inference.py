import os
import sys
import argparse
import json
from typing import List, Dict
from datetime import datetime
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.inferencer import StockPredictor
from train.model_args import ModelArgs
from log.logger import get_logger

# 设置日志
logger = get_logger(__name__, log_file="run_inference.log")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="股票价格预测实时推理")
    
    # 基本参数
    parser.add_argument("--tickers", type=str, default="AAPL",
                       help="要预测的股票代码，多个用逗号分隔，如'AAPL,MSFT,GOOG'")
    parser.add_argument("--model_path", type=str, default="models/SpaceExploreAI_best.pt",
                       help="模型权重路径")
    parser.add_argument("--device", type=str, default=None,
                       help="运行设备，如'cpu', 'cuda', 'mps'，默认自动选择")
    
    # 数据参数
    parser.add_argument("--raw_data_dir", type=str, default="data/raw",
                       help="原始数据目录")
    parser.add_argument("--processed_data_dir", type=str, default="data/processed/inference",
                       help="处理后数据目录")
    parser.add_argument("--feature_groups", type=str, default="time,lag,return,volatility,volume",
                       help="特征组，多个用逗号分隔")
    parser.add_argument("--sequence_length", type=int, default=32,
                       help="序列长度")
    parser.add_argument("--prediction_horizon", type=int, default=2,
                       help="预测周期")
    parser.add_argument("--scaler_type", type=str, default="robust",
                       help="缩放器类型")
    
    # 运行模式参数
    parser.add_argument("--mode", type=str, default="once", choices=["once", "continuous"],
                       help="运行模式，'once'表示单次运行，'continuous'表示持续运行")
    parser.add_argument("--interval", type=int, default=3600,
                       help="连续模式下的更新间隔（秒）")
    parser.add_argument("--output", type=str, default=None,
                       help="输出结果到文件")
    
    return parser.parse_args()

def run_once(predictor: StockPredictor, tickers: List[str], output_file: str = None) -> List[Dict]:
    """
    单次运行预测
    
    参数:
        predictor: 预测器
        tickers: 股票代码列表
        output_file: 输出文件
        
    返回:
        预测结果列表
    """
    logger.info(f"开始预测 {len(tickers)} 只股票: {', '.join(tickers)}")
    
    # 进行预测
    results = predictor.predict_batch(tickers)
    
    # 打印结果
    for result in results:
        if "error" in result:
            logger.error(f"预测 {result.get('ticker', '未知')} 失败: {result['error']}")
        else:
            logger.info(f"预测结果 - 股票: {result['ticker']}, 预测值: {result['prediction']}, 时间: {result['timestamp']}")
    
    # 保存结果到文件
    if output_file:
        try:
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"结果已保存到: {output_file}")
        except Exception as e:
            logger.error(f"保存结果到文件时出错: {str(e)}")
    
    return results

def run_continuous(predictor: StockPredictor, tickers: List[str], interval: int, output_file: str = None):
    """
    持续运行预测
    
    参数:
        predictor: 预测器
        tickers: 股票代码列表
        interval: 更新间隔（秒）
        output_file: 输出文件
    """
    logger.info(f"开始持续预测 {len(tickers)} 只股票: {', '.join(tickers)}")
    logger.info(f"更新间隔: {interval} 秒")
    
    try:
        while True:
            # 获取当前时间
            current_time = datetime.now()
            logger.info(f"开始新一轮预测，当前时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 生成此次预测的输出文件名
            if output_file:
                timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                output_path = output_file.replace(".json", f"_{timestamp}.json")
            else:
                output_path = None
            
            # 进行预测
            results = run_once(predictor, tickers, output_path)
            
            # 休眠直到下次更新
            logger.info(f"休眠 {interval} 秒直到下次更新")
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("收到中断信号，停止预测")
    except Exception as e:
        logger.error(f"持续预测时出错: {str(e)}")
        raise

def load_model_args() -> ModelArgs:
    """加载模型参数"""
    return ModelArgs()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载模型参数
    model_args = load_model_args()
    
    # 解析股票代码
    tickers = args.tickers.split(',')
    
    # 解析特征组
    feature_groups = args.feature_groups.split(',')
    
    # 创建预测器
    predictor = StockPredictor(
        model_path=args.model_path,
        device=args.device,
        feature_groups=feature_groups,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        raw_data_dir=args.raw_data_dir,
        processed_data_dir=args.processed_data_dir,
        scaler_type=args.scaler_type
    )
    
    # 根据运行模式执行预测
    if args.mode == "once":
        run_once(predictor, tickers, args.output)
    else:
        run_continuous(predictor, tickers, args.interval, args.output)

if __name__ == "__main__":
    main() 