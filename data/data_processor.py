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
from data.technical_indicators import TechnicalIndicatorProcessor
from log.logger import get_logger
import sys

logger = get_logger(__name__, log_file="data_processor.log")

# 获取当前文件所在目录的绝对路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 设置默认的原始数据和处理后数据的目录
DEFAULT_RAW_DIR = os.path.join(CURRENT_DIR, "raw", "price_history")
DEFAULT_PROCESSED_DIR = os.path.join(CURRENT_DIR, "processed")

class StockDataProcessor:
    """
    股票数据处理器 - 负责数据清洗、特征工程、数据准备等
    """
    
    def __init__(self, 
                 raw_data_dir: str = DEFAULT_RAW_DIR, 
                 processed_data_dir: str = DEFAULT_PROCESSED_DIR,
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
            feature_groups: 要添加的特征组，可包含: 'technical', 'time', 'lag', 'return', 'volatility', 'all'
            technical_indicators: 具体要添加的技术指标列表，None则添加全部
            
        返回:
            添加了特征的数据框
        """
        logger.info("正在添加特征...")
        
        result_df = df.copy()
        
        if feature_groups is None:
            feature_groups = ['technical', 'time', 'lag', 'return', 'volatility', 'volume']
            
        logger.info(f"要添加的特征组: {feature_groups}")
            
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
                logger.info("已添加时间特征")
                
        # 添加滞后特征
        if 'lag' in feature_groups or 'all' in feature_groups:
            result_df = self._add_lag_features(result_df)
            logger.info("已添加滞后特征")
                
        # 添加收益率特征
        if 'return' in feature_groups or 'all' in feature_groups:
            # 确保包含预测周期2
            periods = [1, 2, 5, 10]
            result_df = self._add_return_features(result_df, periods=periods)
            logger.info("已添加收益率特征")
            
            # 检查是否成功创建了未来收益率列
            future_return_cols = [col for col in result_df.columns if col.startswith('future_return_')]
            if future_return_cols:
                logger.info(f"创建的未来收益率列: {future_return_cols}")
            else:
                logger.error("未能创建任何未来收益率列！")
            
        # 添加波动性特征
        if 'volatility' in feature_groups or 'all' in feature_groups:
            result_df = self._add_volatility_features(result_df)
            logger.info("已添加波动性特征")

        # 添加成交量特征
        if 'volume' in feature_groups or 'all' in feature_groups:
            result_df = self._add_volume_features(result_df)
            logger.info("已添加成交量特征")
            
        # 输出最终特征列
        logger.info(f"最终数据形状: {result_df.shape}, 列数: {len(result_df.columns)}")
        logger.info(f"特征列名: {result_df.columns.tolist()}")
        
        return result_df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加成交量特征
        
        参数:
            df: 输入数据框
            
        返回:
            添加了成交量特征的数据框
            
        注意:
            - volume_change_rate 的第一个值会是 NaN，因为没有前一个值可以计算变化率
            - 我们使用 0 填充这个 NaN 值，因为这表示没有变化
        """
        result_df = df.copy()
        
        # 计算成交量变化率
        result_df['volume_change_rate'] = result_df['volume'].pct_change()
        
        # 将第一个 NaN 值填充为 0，因为第一个数据点没有变化率
        result_df['volume_change_rate'] = result_df['volume_change_rate'].fillna(0)
        
        return result_df
        
        
    
    def _add_lag_features(self, 
                         df: pd.DataFrame, 
                         columns: List[str] = None, 
                         lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
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
                            periods: List[int] = [1, 5, 10, 21]) -> pd.DataFrame:
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
        
        # 检查价格列是否存在
        if price_col not in result_df.columns:
            logger.error(f"价格列 {price_col} 不存在于数据框中，无法计算收益率特征")
            return result_df
            
        # 添加未来收益率（作为预测目标）
        for period in periods:
            future_return_col = f'future_return_{period}d'
            result_df[future_return_col] = (result_df[price_col].shift(-period) / result_df[price_col] - 1) * 100
            
            # 检查创建的未来收益率列
            non_null_count = result_df[future_return_col].count()
            total_count = len(result_df)
            logger.info(f"创建了 {future_return_col} 列: 非空值数量={non_null_count}, 总行数={total_count}, 非空比例={non_null_count/total_count:.2%}")
            
        # 添加历史收益率
        for period in periods:
            past_return_col = f'past_return_{period}d'
            result_df[past_return_col] = (result_df[price_col] / result_df[price_col].shift(period) - 1) * 100
            
            # 检查创建的历史收益率列
            non_null_count = result_df[past_return_col].count()
            total_count = len(result_df)
            logger.info(f"创建了 {past_return_col} 列: 非空值数量={non_null_count}, 总行数={total_count}, 非空比例={non_null_count/total_count:.2%}")
            
        return result_df
    
    def _add_volatility_features(self, 
                               df: pd.DataFrame, 
                               windows: List[int] = [5, 10, 21]) -> pd.DataFrame:
        """
        添加波动性相关特征
        
        参数:
            df: 输入数据框
            windows: 计算窗口大小列表
            
        返回:
            添加了波动性特征的数据框
        """
        result_df = df.copy()
        
        # 检查必要的列是否存在
        required_cols = ['high', 'low', 'close', 'open', 'volume']
        missing_cols = [col for col in required_cols if col not in result_df.columns]
        if missing_cols:
            logger.warning(f"缺少计算波动性特征所需的列: {missing_cols}")
            
        # 1. 当日波动范围 (高-低)/收盘价
        if 'high' in result_df.columns and 'low' in result_df.columns and 'close' in result_df.columns:
            result_df['daily_range'] = (result_df['high'] - result_df['low']) / result_df['close'] * 100
            logger.info("添加了当日波动范围特征: daily_range")
            
        # 2. 当日波动绝对值 (高-低)
        if 'high' in result_df.columns and 'low' in result_df.columns:
            result_df['daily_range_abs'] = result_df['high'] - result_df['low']
            logger.info("添加了当日波动绝对值特征: daily_range_abs")
            
        # 3. 开盘跳空 (开盘价-前日收盘价)/前日收盘价
        if 'open' in result_df.columns and 'close' in result_df.columns:
            result_df['gap'] = (result_df['open'] - result_df['close'].shift(1)) / result_df['close'].shift(1) * 100
            logger.info("添加了开盘跳空特征: gap")
            
        # 4. 收盘价与开盘价的差距 (收盘价-开盘价)/开盘价
        if 'close' in result_df.columns and 'open' in result_df.columns:
            result_df['close_to_open'] = (result_df['close'] - result_df['open']) / result_df['open'] * 100
            logger.info("添加了收盘与开盘差距特征: close_to_open")
            
        # 5. 收盘价与当日最高价的差距 (收盘价-最高价)/最高价
        if 'close' in result_df.columns and 'high' in result_df.columns:
            result_df['close_to_high'] = (result_df['close'] - result_df['high']) / result_df['high'] * 100
            logger.info("添加了收盘与最高价差距特征: close_to_high")
            
        # 6. 收盘价与当日最低价的差距 (收盘价-最低价)/最低价
        if 'close' in result_df.columns and 'low' in result_df.columns:
            result_df['close_to_low'] = (result_df['close'] - result_df['low']) / result_df['low'] * 100
            logger.info("添加了收盘与最低价差距特征: close_to_low")
            
        # 7. 不同窗口的波动率 (标准差)
        if 'close' in result_df.columns:
            for window in windows:
                result_df[f'volatility_{window}d'] = result_df['close'].pct_change().rolling(window=window).std() * 100
                logger.info(f"添加了{window}日波动率特征: volatility_{window}d")
                
        # 8. 不同窗口的平均真实范围 (ATR)
        if 'high' in result_df.columns and 'low' in result_df.columns and 'close' in result_df.columns:
            # 计算前一日收盘价
            close_prev = result_df['close'].shift(1)
            
            # 计算TR的三个部分
            tr1 = result_df['high'] - result_df['low']
            tr2 = (result_df['high'] - close_prev).abs()
            tr3 = (result_df['low'] - close_prev).abs()
            
            # 合并并取每行的最大值，NaN自动处理
            result_df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # 计算不同窗口的ATR
            for window in windows:
                result_df[f'atr_{window}d'] = result_df['tr'].rolling(window=window).mean()
                logger.info(f"添加了{window}日平均真实范围特征: atr_{window}d")
                
            # 删除临时TR列
            result_df.drop('tr', axis=1, inplace=True)
            
        # 9. 价格动量 (当前价格/n日前价格 - 1)
        if 'close' in result_df.columns:
            for window in windows:
                result_df[f'momentum_{window}d'] = (result_df['close'] / result_df['close'].shift(window) - 1) * 100
                logger.info(f"添加了{window}日价格动量特征: momentum_{window}d")
                
        # 10. 成交量变化率
        if 'volume' in result_df.columns:
            for window in windows:
                result_df[f'volume_change_{window}d'] = (result_df['volume'] / result_df['volume'].rolling(window=window).mean() - 1) * 100
                logger.info(f"添加了{window}日成交量变化率特征: volume_change_{window}d")
                
        # 11. 价格与移动平均线的差距
        if 'close' in result_df.columns:
            for window in windows:
                ma_col = f'ma_{window}d'
                result_df[ma_col] = result_df['close'].rolling(window=window).mean()
                result_df[f'close_to_{ma_col}'] = (result_df['close'] / result_df[ma_col] - 1) * 100
                logger.info(f"添加了收盘价与{window}日均线差距特征: close_to_{ma_col}")
                
                # 删除临时MA列
                result_df.drop(ma_col, axis=1, inplace=True)
                
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
                              sequence_length: int = 32,
                              prediction_horizon: int = 2,
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
                    
        logger.info(f"已加载{ticker}:features,target缩放器")
    
    def create_sequence_datasets(self, 
                               data_dict: Dict[str, pd.DataFrame],
                               feature_columns: List[str] = None,
                               target_column: str = 'future_return_5d',
                               sequence_length: int = 32) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
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
        logger.info(f"正在创建序列数据集，序列长度：{sequence_length}，目标列：{target_column}")
        
        # 检查目标列是否存在于数据框中
        for split, df in data_dict.items():
            if target_column in df.columns:
                non_null_count = df[target_column].count()
                total_count = len(df)
                logger.info(f"{split} 集中 {target_column} 列: 非空值数量={non_null_count}, 总行数={total_count}, 非空比例={non_null_count/total_count:.2%}")
            else:
                logger.error(f"{split} 集中不存在 {target_column} 列！")
        
        # 如果未指定特征列，使用所有数值列
        if feature_columns is None:
            # 排除日期、目标列和非数值列
            exclude_cols = ['date', 'ticker']
            # 只添加一次目标列到排除列表
            if target_column is not None and target_column not in exclude_cols:
                exclude_cols.append(target_column)
                
            feature_columns = [col for col in data_dict['train'].columns 
                             if col not in exclude_cols and pd.api.types.is_numeric_dtype(data_dict['train'][col])]
            
            logger.info(f"自动选择了 {len(feature_columns)} 个特征列")
            
        result = {}
        
        for split, df in data_dict.items():
            sequences = []
            targets = []
            
            # 检查目标列是否存在
            if target_column not in df.columns:
                logger.error(f"{split} 集中不存在目标列 {target_column}，无法创建目标值")
                continue
                
            for i in range(len(df) - sequence_length):
                # 提取特征序列
                seq = df.iloc[i:i+sequence_length][feature_columns].values
                sequences.append(seq)
                
                # 提取目标值（序列末尾的未来收益率）
                target = df.iloc[i+sequence_length-1][target_column]
                # 确保目标值不是 NaN
                if pd.notna(target):
                    targets.append(target)
                else:
                    # 如果目标值是 NaN，使用 0 填充（或者其他合适的默认值）
                    targets.append(0.0)
                    logger.debug(f"在 {split} 集的索引 {i+sequence_length-1} 处发现 NaN 目标值，已用 0 填充")
            
            if sequences:
                result[split] = (np.array(sequences), np.array(targets))
                logger.info(f"{split} 集创建了 {len(sequences)} 个序列，目标值数量: {len(targets)}")
                
                # 检查序列和目标的长度是否匹配
                if len(sequences) != len(targets):
                    logger.error(f"{split} 集的序列数量 ({len(sequences)}) 与目标值数量 ({len(targets)}) 不匹配！")
            else:
                logger.warning(f"{split} 集没有足够的数据创建序列")
                
        return result
    
    def process_stock_pipeline(self, 
                             ticker: str, 
                             source: str = 'yahoo',
                             test_size: float = 0.1,
                             val_size: float = 0.1,
                             sequence_length: int = 32,
                             prediction_horizon: int = 2,
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
        logger.info(f"使用目标列: {target_column}")
        
        # 检查目标列是否存在
        if target_column not in df.columns:
            logger.error(f"目标列 {target_column} 不存在于处理后的数据中！")
            # 检查所有可能的未来收益率列
            future_return_cols = [col for col in df.columns if col.startswith('future_return_')]
            if future_return_cols:
                logger.info(f"数据中存在的未来收益率列: {future_return_cols}")
                # 使用第一个可用的未来收益率列作为目标
                target_column = future_return_cols[0]
                logger.info(f"改用 {target_column} 作为目标列")
            else:
                logger.error(f"数据中不存在任何未来收益率列，无法继续处理")
                return None
        
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
        
        # 检查序列数据
        for split, (X, y) in sequences.items():
            logger.info(f"{split} 序列数据: X形状={X.shape}, y形状={y.shape}")
            if len(y) == 0:
                logger.error(f"{split} 目标值数组为空！")
        
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
        # 为每个ticker创建目录
        ticker_dir = os.path.join(self.processed_data_dir, ticker)
        
        # 保存原始分割到对应目录
        for split, df in splits.items():
            # 根据split类型选择对应的目录
            if split == 'train':
                save_dir = os.path.join(self.processed_data_dir, "train")
            elif split == 'val':
                save_dir = os.path.join(self.processed_data_dir, "eval")
            else:  # test
                save_dir = os.path.join(self.processed_data_dir, "test")
            
            os.makedirs(save_dir, exist_ok=True)
            file_path = os.path.join(save_dir, f"{ticker}_{split}_data.csv")
            df.to_csv(file_path, index=False)
            
        # 保存缩放后的分割到对应目录
        for split, df in scaled_splits.items():
            # 根据split类型选择对应的目录
            if split == 'train':
                save_dir = os.path.join(self.processed_data_dir, "train")
            elif split == 'val':
                save_dir = os.path.join(self.processed_data_dir, "eval")
            else:  # test
                save_dir = os.path.join(self.processed_data_dir, "test")
            
            os.makedirs(save_dir, exist_ok=True)
            file_path = os.path.join(save_dir, f"{ticker}_{split}_scaled_data.csv")
            df.to_csv(file_path, index=False)
            
        # 保存序列数据到对应目录
        for split, (X, y) in sequences.items():
            # 根据split类型选择对应的目录
            if split == 'train':
                save_dir = os.path.join(self.processed_data_dir, "train")
            elif split == 'val':
                save_dir = os.path.join(self.processed_data_dir, "eval")
            else:  # test
                save_dir = os.path.join(self.processed_data_dir, "test")
            
            os.makedirs(save_dir, exist_ok=True)
            file_path = os.path.join(save_dir, f"{ticker}_{split}_sequences.npz")
            np.savez(file_path, X=X, y=y)
            
        logger.info(f"已保存处理后的数据到对应目录")
    
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
            pin_memory=False
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