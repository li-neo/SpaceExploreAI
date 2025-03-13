import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from technical_indicators import TechnicalIndicatorProcessor
from log.logger import get_logger
import sys

logger = get_logger(__name__, log_file="data_processor.log")


class StockDataProcessor:
    """
    股票数据处理器 - 负责数据清洗、特征工程、数据准备等
    """
    
    def __init__(self, 
                 raw_data_dir: str = "../data/raw", 
                 processed_data_dir: str = "../data/processed",
                 scaler_type: str = "robust"):
        """
        初始化股票数据处理器
        
        参数:
            raw_data_dir: 原始数据存储目录
            processed_data_dir: 处理后数据存储目录
            scaler_type: 数据标准化方式，可选 'standard', 'minmax', 'robust'
        """
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.scaler_type = scaler_type
        self.tech_processor = TechnicalIndicatorProcessor()
        
        # 确保处理后数据目录存在
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.processed_data_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.processed_data_dir, "eval"), exist_ok=True)
        os.makedirs(os.path.join(self.processed_data_dir, "test"), exist_ok=True)
        os.makedirs(os.path.join(self.processed_data_dir, "scalers"), exist_ok=True)
        
        # 创建特征缩放器
        self.scalers = {}
        
    def _get_scaler(self, name: str = "price"):
        """根据配置创建对应的缩放器"""
        if self.scaler_type == "standard":
            return StandardScaler()
        elif self.scaler_type == "minmax":
            return MinMaxScaler(feature_range=(-1, 1))
        else:  # robust scaler by default
            return RobustScaler()
            
    def clean_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗股票数据
        
        参数:
            df: 原始股票数据DataFrame
            
        返回:
            清洗后的DataFrame
        """
        logger.info("正在清洗股票数据...")
        
        # 复制数据框
        clean_df = df.copy()

        # 处理多级索引列名，只保留第一级
        if isinstance(clean_df.columns, pd.MultiIndex):
            clean_df.columns = clean_df.columns.get_level_values(0)
        
        # 确保列名都是小写的
        clean_df.columns = [col.lower() for col in clean_df.columns]
        
        # 将日期列转换为日期时间类型
        if 'date' in clean_df.columns and not pd.api.types.is_datetime64_any_dtype(clean_df['date']):
            clean_df['date'] = pd.to_datetime(clean_df['date'])
            
        # 按日期排序
        if 'date' in clean_df.columns:
            clean_df.sort_values('date', inplace=True)
            
        # 处理缺失值
        # 对于OHLC价格数据，使用前向填充
        for col in ['open', 'high', 'low', 'close', 'adj_close']:
            if col in clean_df.columns:
                clean_df[col] = clean_df[col].ffill()
                
        # 对于成交量等数据，可以用0或均值填充
        if 'volume' in clean_df.columns:
            clean_df['volume'] = clean_df['volume'].fillna(clean_df['volume'].mean())
            
        # 移除仍然有缺失值的行
        clean_df.dropna(inplace=True)

        if clean_df.empty:
            logger.error("数据清洗后数据为空")
            sys.exit(1)
        
        # 移除重复的行
        clean_df.drop_duplicates(inplace=True)

        if clean_df.empty:
            logger.error("移除重复行后数据为空")
            sys.exit(1)
        
        # 移除异常值 (可选，根据需要打开此功能)
        # clean_df = self._remove_outliers(clean_df)
        
        return clean_df
    
    def _remove_outliers(self, 
                        df: pd.DataFrame, 
                        columns: List[str] = None, 
                        z_threshold: float = 3.0) -> pd.DataFrame:
        """
        使用Z-score方法移除异常值
        
        参数:
            df: 输入数据框
            columns: 需要处理的列，如果为None则处理所有数值列
            z_threshold: Z-score阈值，超过此值视为异常值
            
        返回:
            移除异常值后的数据框
        """
        result_df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=np.number).columns.tolist()
            
        for column in columns:
            if column in result_df.columns:
                mean = result_df[column].mean()
                std = result_df[column].std()
                z_scores = np.abs((result_df[column] - mean) / std)
                result_df = result_df[z_scores < z_threshold]
                
        return result_df
    
    def add_features(self, 
                    df: pd.DataFrame, 
                    feature_groups: List[str] = None,
                    technical_indicators: List[str] = None) -> pd.DataFrame:
        """
        添加特征
        
        参数:
            df: 输入数据框
            feature_groups: 要添加的特征组，可包含: 'technical', 'time', 'lag', 'return', 'all'
            technical_indicators: 具体要添加的技术指标列表，None则添加全部
            
        返回:
            添加了特征的数据框
        """
        logger.info("正在添加特征...")
        
        result_df = df.copy()
        
        if feature_groups is None:
            feature_groups = ['technical', 'time', 'lag', 'return']
            
        # 添加技术指标(不添加任何指标， 从原始数据上开始训练)
        # if 'technical' in feature_groups or 'all' in feature_groups:
        #     if technical_indicators is None:
        #         result_df = self.tech_processor.calculate_all_indicators(result_df)
        #     else:
        #         result_df = self.tech_processor.process_stock_data(result_df, technical_indicators)
                
        # 添加时间特征
        if ('time' in feature_groups or 'all' in feature_groups) and 'date' in result_df.columns:
            if 'day_of_week' not in result_df.columns:  # 避免重复添加
                result_df = self.tech_processor.add_time_features(result_df)
                
        # 添加滞后特征
        if 'lag' in feature_groups or 'all' in feature_groups:
            result_df = self._add_lag_features(result_df)
                
        # 添加收益率特征
        if 'return' in feature_groups or 'all' in feature_groups:
            result_df = self._add_return_features(result_df)
            
        return result_df
    
    def _add_lag_features(self, 
                         df: pd.DataFrame, 
                         columns: List[str] = None, 
                         lags: List[int] = [1, 2, 3, 5, 10, 21]) -> pd.DataFrame:
        """
        添加滞后特征
        
        参数:
            df: 输入数据框
            columns: 需要添加滞后特征的列，默认为价格和成交量
            lags: 滞后周期列表
            
        返回:
            添加了滞后特征的数据框
        """
        result_df = df.copy()
        
        if columns is None:
            columns = ['close', 'high', 'low', 'volume']
            columns = [col for col in columns if col in result_df.columns]
            
        for col in columns:
            for lag in lags:
                result_df[f'{col}_lag_{lag}'] = result_df[col].shift(lag)
                
        return result_df
    
    def _add_return_features(self, 
                            df: pd.DataFrame, 
                            price_col: str = 'close', 
                            periods: List[int] = [1, 5, 10, 21, 63]) -> pd.DataFrame:
        """
        添加收益率特征
        
        参数:
            df: 输入数据框
            price_col: 用于计算收益率的价格列
            periods: 计算收益率的周期列表
            
        返回:
            添加了收益率特征的数据框
        """
        result_df = df.copy()
        
        # 添加未来收益率（作为预测目标）
        for period in periods:
            result_df[f'future_return_{period}d'] = (result_df[price_col].shift(-period) / result_df[price_col] - 1) * 100
            
        # 添加历史收益率
        for period in periods:
            result_df[f'past_return_{period}d'] = (result_df[price_col] / result_df[price_col].shift(period) - 1) * 100
            
        return result_df
    
    def load_and_process_stock_data(self, 
                                   ticker: str, 
                                   source: str = 'yahoo', 
                                   feature_groups: List[str] = None) -> pd.DataFrame:
        """
        加载并处理单个股票的数据
        
        参数:
            ticker: 股票代码
            source: 数据来源，'yahoo' 或 'alphavantage'
            feature_groups: 要添加的特征组
            
        返回:
            处理后的股票数据
        """
        # 构建股票数据文件路径
        if source == 'yahoo':
            ticker_files = [f for f in os.listdir(os.path.join(self.raw_data_dir, "price_history", ticker)) 
                          if f.endswith('.csv') and 'yahoo' in f]
        else:
            ticker_files = [f for f in os.listdir(os.path.join(self.raw_data_dir, "price_history", ticker)) 
                          if f.endswith('.csv') and 'alphavantage' in f]
                          
        if not ticker_files:
            logger.error(f"未找到{ticker}的数据文件")
            return None
            
        # 获取最新的数据文件
        ticker_file = sorted(ticker_files)[-1]
        file_path = os.path.join(self.raw_data_dir, "price_history", ticker, ticker_file)
        
        # 读取数据
        df = pd.read_csv(file_path)
        
        # 清洗数据
        clean_df = self.clean_stock_data(df)
        
        # 添加特征
        processed_df = self.add_features(clean_df, feature_groups)
        
        return processed_df
    
    def prepare_dataset_splits(self, 
                              df: pd.DataFrame,
                              test_size: float = 0.1,
                              val_size: float = 0.1,
                              sequence_length: int = 60,
                              prediction_horizon: int = 5,
                              target_column: str = 'future_return_5d') -> Dict[str, pd.DataFrame]:
        """
        准备训练、验证和测试数据集
        
        参数:
            df: 输入数据框
            test_size: 测试集比例
            val_size: 验证集比例
            sequence_length: 序列长度（用于时间序列预测）
            prediction_horizon: 预测周期
            target_column: 目标列名
            
        返回:
            包含训练、验证和测试数据集的字典
        """
        # 确保数据按日期排序
        if 'date' in df.columns:
            df = df.sort_values('date')

        df.fillna(0, inplace=True)
            
        # 移除有缺失值的行
        # df.dropna(inplace=True)
        # 将df输出到csv
        df.to_csv("data/processed/data_before_split.csv", index=False)
        if df.empty:
            logger.error("移除缺失值后数据为空, 请检查数据中哪些是空值")
            sys.exit(1)
        
        # 分割数据集
        n = len(df)
        test_indices = int(n * (1 - test_size))
        val_indices = int(test_indices * (1 - val_size))
        
        train_df = df.iloc[:val_indices].copy()
        val_df = df.iloc[val_indices:test_indices].copy()
        test_df = df.iloc[test_indices:].copy()
        
        logger.info(f"数据集分割: 训练集 {len(train_df)} 行, 验证集 {len(val_df)} 行, 测试集 {len(test_df)} 行")
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
    
    def scale_features(self, 
                      train_df: pd.DataFrame, 
                      val_df: pd.DataFrame = None, 
                      test_df: pd.DataFrame = None,
                      feature_columns: List[str] = None,
                      target_column: str = 'future_return_5d',
                      fit_on_train: bool = True) -> Dict[str, pd.DataFrame]:
        """
        缩放特征
        
        参数:
            train_df: 训练数据集
            val_df: 验证数据集
            test_df: 测试数据集
            feature_columns: 需要缩放的特征列
            target_column: 目标列
            fit_on_train: 是否仅基于训练集拟合缩放器
            
        返回:
            包含缩放后数据的字典
        """
        logger.info("正在缩放特征...")
        
        if train_df.empty:
            logger.warning("Training DataFrame is empty. Check data processing steps.")
            return None
        
        # 如果未指定特征列，使用所有数值列
        if feature_columns is None:
            # 排除日期、目标列和非数值列
            exclude_cols = ['date', 'ticker', target_column]
            if target_column is not None:
                exclude_cols.append(target_column)
                
            feature_columns = [col for col in train_df.columns 
                             if col not in exclude_cols and pd.api.types.is_numeric_dtype(train_df[col])]
            
        # 创建特征缩放器
        self.scalers['features'] = self._get_scaler("features")
        
        # 拟合并转换训练集
        train_scaled = train_df.copy()
        train_scaled[feature_columns] = self.scalers['features'].fit_transform(train_df[feature_columns])
        
        result = {'train': train_scaled}
        
        # 转换验证集
        if val_df is not None:
            val_scaled = val_df.copy()
            val_scaled[feature_columns] = self.scalers['features'].transform(val_df[feature_columns])
            result['val'] = val_scaled
            
        # 转换测试集
        if test_df is not None:
            test_scaled = test_df.copy()
            test_scaled[feature_columns] = self.scalers['features'].transform(test_df[feature_columns])
            result['test'] = test_scaled
            
        # 如果目标列存在且为数值，也缩放目标
        if target_column in train_df.columns and pd.api.types.is_numeric_dtype(train_df[target_column]):
            self.scalers['target'] = self._get_scaler("target")
            
            # 重塑以适应缩放器的输入要求
            target_train = train_df[target_column].values.reshape(-1, 1)
            result['train'][target_column] = self.scalers['target'].fit_transform(target_train).flatten()
            
            if val_df is not None and target_column in val_df.columns:
                target_val = val_df[target_column].values.reshape(-1, 1)
                result['val'][target_column] = self.scalers['target'].transform(target_val).flatten()
                
            if test_df is not None and target_column in test_df.columns:
                target_test = test_df[target_column].values.reshape(-1, 1)
                result['test'][target_column] = self.scalers['target'].transform(target_test).flatten()
                
        return result
    
    def save_scalers(self, ticker: str = None):
        """
        保存特征缩放器
        
        参数:
            ticker: 股票代码，用于命名文件
        """
        scaler_dir = os.path.join(self.processed_data_dir, "scalers")
        os.makedirs(scaler_dir, exist_ok=True)
        
        prefix = ticker + "_" if ticker else ""
        
        for name, scaler in self.scalers.items():
            file_path = os.path.join(scaler_dir, f"{prefix}{name}_scaler.pkl")
            with open(file_path, 'wb') as f:
                pickle.dump(scaler, f)
                
        logger.info(f"已保存缩放器到 {scaler_dir}")
    
    def load_scalers(self, ticker: str = None):
        """
        加载特征缩放器
        
        参数:
            ticker: 股票代码，用于查找文件
        """
        scaler_dir = os.path.join(self.processed_data_dir, "scalers")
        
        prefix = ticker + "_" if ticker else ""
        
        for name in ['features', 'target']:
            file_path = os.path.join(scaler_dir, f"{prefix}{name}_scaler.pkl")
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    self.scalers[name] = pickle.load(f)
                    
        logger.info(f"已加载缩放器")
    
    def create_sequence_datasets(self, 
                               data_dict: Dict[str, pd.DataFrame],
                               feature_columns: List[str] = None,
                               target_column: str = 'future_return_5d',
                               sequence_length: int = 60) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        创建序列数据集
        
        参数:
            data_dict: 包含训练、验证和测试数据集的字典
            feature_columns: 特征列
            target_column: 目标列
            sequence_length: 序列长度
            
        返回:
            包含序列数据的字典
        """
        logger.info(f"正在创建序列数据集，序列长度：{sequence_length}")
        
        # 如果未指定特征列，使用所有数值列
        if feature_columns is None:
            # 排除日期、目标列和非数值列
            exclude_cols = ['date', 'ticker', target_column]
            if target_column is not None:
                exclude_cols.append(target_column)
                
            feature_columns = [col for col in data_dict['train'].columns 
                             if col not in exclude_cols and pd.api.types.is_numeric_dtype(data_dict['train'][col])]
            
        result = {}
        
        for split, df in data_dict.items():
            sequences = []
            targets = []
            
            for i in range(len(df) - sequence_length):
                # 提取特征序列
                seq = df.iloc[i:i+sequence_length][feature_columns].values
                sequences.append(seq)
                
                # 提取目标值（序列末尾的未来收益率）
                if target_column in df.columns:
                    target = df.iloc[i+sequence_length-1][target_column]
                    targets.append(target)
            
            if sequences:
                result[split] = (np.array(sequences), np.array(targets))
                logger.info(f"{split} 集创建了 {len(sequences)} 个序列")
            else:
                logger.warning(f"{split} 集没有足够的数据创建序列")
                
        return result
    
    def process_stock_pipeline(self, 
                             ticker: str, 
                             source: str = 'yahoo',
                             test_size: float = 0.1,
                             val_size: float = 0.1,
                             sequence_length: int = 60,
                             prediction_horizon: int = 5,
                             feature_groups: List[str] = None,
                             save_data: bool = True) -> Dict:
        """
        完整的股票数据处理流水线
        
        参数:
            ticker: 股票代码
            source: 数据来源
            test_size: 测试集比例
            val_size: 验证集比例
            sequence_length: 序列长度
            prediction_horizon: 预测周期
            feature_groups: 要添加的特征组
            save_data: 是否保存处理后的数据
            
        返回:
            处理结果
        """
        logger.info(f"开始处理 {ticker} 的数据")
        
        # 1. 加载并处理数据
        df = self.load_and_process_stock_data(ticker, source, feature_groups)
        if df is None:
            return None
            
        # 2. 准备数据集分割
        target_column = f'future_return_{prediction_horizon}d'
        splits = self.prepare_dataset_splits(
            df, 
            test_size=test_size, 
            val_size=val_size,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            target_column=target_column
        )
        
        # 3. 缩放特征
        scaled_splits = self.scale_features(
            splits['train'], 
            splits['val'], 
            splits['test'],
            target_column=target_column
        )
        
        # 4. 创建序列数据集
        sequences = self.create_sequence_datasets(
            scaled_splits,
            target_column=target_column,
            sequence_length=sequence_length
        )
        
        # 5. 保存数据和缩放器
        if save_data:
            self.save_processed_data(ticker, splits, scaled_splits, sequences)
            self.save_scalers(ticker)
            
        # 返回处理结果
        return {
            'raw_data': df,
            'splits': splits,
            'scaled_splits': scaled_splits,
            'sequences': sequences
        }
    
    def save_processed_data(self, 
                           ticker: str, 
                           splits: Dict[str, pd.DataFrame],
                           scaled_splits: Dict[str, pd.DataFrame],
                           sequences: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        """
        保存处理后的数据
        
        参数:
            ticker: 股票代码
            splits: 数据分割
            scaled_splits: 缩放后的数据分割
            sequences: 序列数据
        """
        ticker_dir = os.path.join(self.processed_data_dir, ticker)
        os.makedirs(ticker_dir, exist_ok=True)
        
        # 保存原始分割
        for split, df in splits.items():
            file_path = os.path.join(ticker_dir, f"{split}_data.csv")
            df.to_csv(file_path, index=False)
            
        # 保存缩放后的分割
        for split, df in scaled_splits.items():
            file_path = os.path.join(ticker_dir, f"{split}_scaled_data.csv")
            df.to_csv(file_path, index=False)
            
        # 保存序列数据
        for split, (X, y) in sequences.items():
            file_path = os.path.join(ticker_dir, f"{split}_sequences.npz")
            np.savez(file_path, X=X, y=y)
            
        logger.info(f"已保存处理后的数据到 {ticker_dir}")
    
    def batch_process_stocks(self, 
                            tickers: List[str], 
                            source: str = 'yahoo',
                            **kwargs):
        """
        批量处理多只股票数据
        
        参数:
            tickers: 股票代码列表
            source: 数据来源
            **kwargs: 其他参数传递给 process_stock_pipeline
        """
        results = {}
        
        for ticker in tqdm(tickers, desc="处理股票数据"):
            result = self.process_stock_pipeline(ticker, source, **kwargs)
            results[ticker] = result
            
        return results


class StockDataset(Dataset):
    """用于PyTorch的股票数据集类"""
    
    def __init__(self, X, y):
        """
        初始化数据集
        
        参数:
            X: 特征数据
            y: 目标数据
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        """返回数据集长度"""
        return len(self.X)
        
    def __getitem__(self, idx):
        """获取数据集项"""
        return self.X[idx], self.y[idx]


def create_dataloaders(sequences: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                       batch_size: int = 32,
                       num_workers: int = 4) -> Dict[str, DataLoader]:
    """
    创建PyTorch DataLoader
    
    参数:
        sequences: 序列数据
        batch_size: 批量大小
        num_workers: 数据加载线程数
        
    返回:
        包含DataLoader的字典
    """
    dataloaders = {}
    
    for split, (X, y) in sequences.items():
        dataset = StockDataset(X, y)
        shuffle = (split == 'train')  # 只有训练集需要打乱
        
        dataloaders[split] = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
        
    return dataloaders


# 使用示例
if __name__ == "__main__":
    import yfinance as yf
    
    # 获取股票数据
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    # 使用yfinance下载数据，作为示例
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    data.reset_index(inplace=True)  # 将日期从索引转为列
    data['ticker'] = ticker  # 添加ticker列
    
    # 在数据下载后检查数据是否为空
    if data.empty:
        logger.error(f"未能获取到 {ticker} 的数据")
        sys.exit(1)

    logger.info(f"\n原始数据预览:")
    
    logger.info("-" * 50)
    logger.info(f"数据形状: {data.shape}")
    logger.info(f"数据列名: {data.columns.tolist()}")
    logger.info(f"数据预览:\n{data.head()}")
    logger.info("-" * 50)

    # 创建数据处理器
    processor = StockDataProcessor(
        raw_data_dir="./SpaceExploreAI/data/raw", 
        processed_data_dir="./SpaceExploreAI/data/processed"
    )
    
    # 清洗数据
    clean_data = processor.clean_stock_data(data)
    
    # 在数据清洗后检查数据是否为空
    if clean_data.empty:
        logger.error("清洗后的数据为空，请检查数据清洗步骤。")
        sys.exit(1)

    # 添加特征
    processed_data = processor.add_features(clean_data, feature_groups=['technical', 'time', 'lag', 'return'])
    if processed_data.empty:
        logger.error("添加特征后数据为空")
        sys.exit(1)

    # 准备数据集分割
    splits = processor.prepare_dataset_splits(processed_data)
    
    # 在数据分割后检查数据集是否为空
    if splits['train'].empty or splits['val'].empty or splits['test'].empty:
        logger.error("数据集分割后某个数据集为空，请检查数据分割步骤。")
        sys.exit(1)

    # 缩放特征
    scaled_splits = processor.scale_features(
        splits['train'], 
        splits['val'], 
        splits['test'],
        target_column='future_return_5d'
    )
    
    # 创建序列数据集
    sequences = processor.create_sequence_datasets(scaled_splits, target_column='future_return_5d', sequence_length=16)
    
    # 创建DataLoader
    dataloaders = create_dataloaders(sequences, batch_size=32)
    
    # 查看结果
    for split, dataloader in dataloaders.items():
        X_batch, y_batch = next(iter(dataloader))
        print(f"{split} batch shape: {X_batch.shape}, {y_batch.shape}") 