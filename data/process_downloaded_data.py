import os
import pandas as pd
from tqdm import tqdm
import sys
from data_processor import StockDataProcessor, create_dataloaders
from log.logger import get_logger
from dataclasses import dataclass
from typing import List, Optional
# 设置日志
logger = get_logger(__name__, log_file="process_downloaded_data.log")

# 获取当前文件所在目录的绝对路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 设置默认的原始数据和处理后数据的目录
DEFAULT_RAW_DIR = os.path.join(CURRENT_DIR, "raw", "price_history")
DEFAULT_PROCESSED_DIR = os.path.join(CURRENT_DIR, "processed")

os.makedirs(DEFAULT_PROCESSED_DIR, exist_ok=True)

class DataArgs:
    """数据处理参数配置类"""
    raw_dir: str = DEFAULT_RAW_DIR
    processed_dir: str = DEFAULT_PROCESSED_DIR 
    
    # 数据分割配置
    test_size: float = 0.1
    val_size: float = 0.1
    
    # 序列配置
    seq_length: int = 60
    pred_horizon: int = 5
    
    # 特征配置
    feature_groups: List[str] = ['technical', 'time', 'lag', 'return']
    technical_indicators: Optional[List[str]] = ['SMA', 'EMA', 'RSI', 'MACD', 'BB', 'ATR', 'OBV']
    
    # 数据加载配置
    batch_size: int = 32
    num_workers: int = 2
    
    # 数据缩放配置
    scaler: str = "robust"
    
    # 数据源配置
    source: str = "yahoo"

    tickers: Optional[List[str]] = None
    

def process_all_downloaded_stocks(
    raw_data_dir=DEFAULT_RAW_DIR,
    processed_data_dir=DEFAULT_PROCESSED_DIR,
    test_size=0.1,
    val_size=0.1,
    sequence_length=60,
    prediction_horizon=5,
    feature_groups=None,
    batch_size=32,
    scaler_type="robust"
):
    """
    批量处理所有下载的股票数据
    
    参数:
        raw_data_dir: 原始数据目录
        processed_data_dir: 处理后数据存储目录
        test_size: 测试集比例
        val_size: 验证集比例
        sequence_length: 序列长度
        prediction_horizon: 预测周期
        feature_groups: 要添加的特征组，默认为 ['technical', 'time', 'lag', 'return']
        batch_size: 批处理大小
        scaler_type: 数据标准化方式
    
    返回:
        处理结果的字典
    """
    if feature_groups is None:
        feature_groups = ['technical', 'time', 'lag', 'return']
    
    # 创建数据处理器
    processor = StockDataProcessor(
        raw_data_dir=raw_data_dir,
        processed_data_dir=processed_data_dir,
        scaler_type=scaler_type
    )
    
    # 获取所有股票目录
    try:
        tickers = [d for d in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, d))]
        logger.info(f"找到 {len(tickers)} 个股票目录")
    except FileNotFoundError:
        logger.error(f"找不到原始数据目录: {raw_data_dir}")
        return {}
    
    if not tickers:
        logger.warning("没有找到任何股票数据目录")
        return {}
    
    # 处理结果
    results = {}
    
    # 批量处理每个股票
    for ticker in tqdm(tickers, desc="处理股票数据"):
        logger.info(f"开始处理 {ticker} 的数据")
        
        try:
            # 查找该股票的所有CSV文件
            ticker_dir = os.path.join(raw_data_dir, ticker)
            csv_files = [f for f in os.listdir(ticker_dir) if f.endswith('.csv')]
            
            if not csv_files:
                logger.warning(f"{ticker} 目录下没有找到CSV文件")
                continue
                
            # 选择最新的CSV文件
            latest_file = sorted(csv_files)[-1]
            file_path = os.path.join(ticker_dir, latest_file)
            
            # 读取数据
            df = pd.read_csv(file_path)
            
            # 检查数据是否为空
            if df.empty:
                logger.warning(f"{ticker} 的数据为空")
                continue
                
            # 添加ticker列
            df['ticker'] = ticker
            
            # 清洗数据
            clean_df = processor.clean_stock_data(df)
            
            # 检查清洗后的数据是否为空
            if clean_df.empty:
                logger.warning(f"{ticker} 清洗后的数据为空")
                continue
                
            # 添加特征
            processed_df = processor.add_features(clean_df, feature_groups)
            
            # 准备数据集分割
            target_column = f'future_return_{prediction_horizon}d'
            splits = processor.prepare_dataset_splits(
                processed_df,
                test_size=test_size,
                val_size=val_size,
                sequence_length=sequence_length,
                prediction_horizon=prediction_horizon,
                target_column=target_column
            )
            
            # 缩放特征
            scaled_splits = processor.scale_features(
                splits['train'],
                splits['val'],
                splits['test'],
                target_column=target_column
            )
            
            # 创建序列数据集
            sequences = processor.create_sequence_datasets(
                scaled_splits,
                target_column=target_column,
                sequence_length=sequence_length
            )
            
            # 保存处理后的数据和缩放器
            processor.save_processed_data(ticker, splits, scaled_splits, sequences)
            processor.save_scalers(ticker)
            
            # 创建DataLoader
            dataloaders = create_dataloaders(sequences, batch_size=batch_size)
            
            # 记录结果
            results[ticker] = {
                'splits': splits,
                'scaled_splits': scaled_splits,
                'sequences': sequences,
                'dataloaders': dataloaders
            }
            
            logger.info(f"{ticker} 处理完成")
            
        except Exception as e:
            logger.error(f"处理 {ticker} 时出错: {str(e)}", exc_info=True)
    
    logger.info(f"所有股票处理完成，成功处理 {len(results)} 个股票")
    return results

def process_specific_stocks(
    tickers,
    raw_data_dir=DEFAULT_RAW_DIR,
    processed_data_dir=DEFAULT_PROCESSED_DIR,
    **kwargs
):
    """
    处理指定的股票列表
    
    参数:
        tickers: 要处理的股票代码列表
        raw_data_dir: 原始数据目录
        processed_data_dir: 处理后数据存储目录
        **kwargs: 其他参数传递给 process_all_downloaded_stocks
    """
    # 创建数据处理器
    processor = StockDataProcessor(
        raw_data_dir=raw_data_dir,
        processed_data_dir=processed_data_dir,
        scaler_type=kwargs.get('scaler_type', 'robust')
    )
    
    # 批量处理指定的股票
    return processor.batch_process_stocks(
        tickers=tickers,
        source='yahoo',  # 假设数据来源是yahoo
        test_size=kwargs.get('test_size', 0.1),
        val_size=kwargs.get('val_size', 0.1),
        sequence_length=kwargs.get('sequence_length', 60),
        prediction_horizon=kwargs.get('prediction_horizon', 5),
        feature_groups=kwargs.get('feature_groups', ['technical', 'time', 'lag', 'return']),
        save_data=True
    )

if __name__ == "__main__":
    args = DataArgs()
    
    # 设置特征组
    feature_groups = ['technical', 'time', 'lag', 'return']
    
    if args.tickers:
        # 处理指定的股票
        logger.info(f"开始处理指定的 {len(args.tickers)} 只股票")
        results = process_specific_stocks(
            tickers=args.tickers,
            raw_data_dir=args.raw_dir,
            processed_data_dir=args.processed_dir,
            test_size=args.test_size,
            val_size=args.val_size,
            sequence_length=args.seq_length,
            prediction_horizon=args.pred_horizon,
            feature_groups=feature_groups,
            scaler_type=args.scaler
        )
    else:
        # 处理所有下载的股票
        logger.info("开始处理所有下载的股票")
        results = process_all_downloaded_stocks(
            raw_data_dir=args.raw_dir,
            processed_data_dir=args.processed_dir,
            test_size=args.test_size,
            val_size=args.val_size,
            sequence_length=args.seq_length,
            prediction_horizon=args.pred_horizon,
            feature_groups=feature_groups,
            batch_size=args.batch_size,
            scaler_type=args.scaler
        )
    
    # 输出处理结果统计
    successful_tickers = list(results.keys())
    logger.info(f"成功处理的股票: {len(successful_tickers)}")
    if successful_tickers:
        logger.info(f"股票列表: {successful_tickers}") 