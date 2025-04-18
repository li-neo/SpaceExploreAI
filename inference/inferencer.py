import os
import sys
import torch
import numpy as np
import pandas as pd
import argparse
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from log.logger import get_logger
import logging
import math

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目内模块
from model.transformer import StockPricePredictor
from data.download_data import download_yahoo_finance_data
from data.process_downloaded_data import DataArgs, process_specific_stocks
from train.model_args import ModelArgs

# 设置日志
logger = get_logger(__name__, log_file="inference.log")

class StockPredictor:
    """股票价格预测器，用于实时推理"""
    
    def __init__(
        self, 
        model_path: str = "models/SpaceExploreAI_best.pt",
        device: str = None,
        feature_groups: List[str] = ['time', 'lag', 'return', 'volatility', 'volume'],
        sequence_length: int = 32,
        prediction_horizon: int = 2,
        raw_data_dir: str = "data/raw",
        processed_data_dir: str = "data/processed/inference",
        scaler_type: str = "robust"
    ):
        """
        初始化股票预测器
        
        参数:
            model_path: 模型权重路径
            device: 运行设备
            feature_groups: 特征组
            sequence_length: 序列长度
            prediction_horizon: 预测周期
            raw_data_dir: 原始数据目录
            processed_data_dir: 处理后数据目录
            scaler_type: 缩放器类型
        """
        self.model_path = model_path
        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else 
            "mps" if torch.backends.mps.is_available() else 
            "cpu"
        )
        self.feature_groups = feature_groups
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.scaler_type = scaler_type
        self.model = None
        self.logger = logger  # 使用全局的logger
        
        # 创建目录
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        self.logger.info(f"使用设备: {self.device}")
        self._load_model()
        
    def _load_model(self):
        """加载模型"""
        self.logger.info(f"正在加载模型: {self.model_path}")
        try:
            self.predictor = StockPricePredictor.load(self.model_path, device=self.device)
            self.logger.info("模型加载成功")
        except Exception as e:
            self.logger.error(f"加载模型时出错: {str(e)}")
            raise
    
    def _download_latest_data(self, ticker: str, lookback_days: int = 180, interval: str = "1d") -> bool:
        """
        下载最新的股票数据
        
        参数:
            ticker: 股票代码
            lookback_days: 回溯天数
            interval: 数据间隔
            
        返回:
            是否成功下载
        """
        self.logger.info(f"下载股票 {ticker} 的最新数据")
        
        # 计算开始日期
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        
        # 调用下载函数
        return download_yahoo_finance_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            output_dir=self.raw_data_dir,
            interval=interval
        )
    
    def _process_data(self, ticker: str) -> Optional[np.ndarray]:
        """
        处理股票数据
        
        参数:
            ticker: 股票代码
            
        返回:
            处理后的特征数据，如果处理失败则返回None
        """
        self.logger.info(f"处理股票 {ticker} 的数据")
        
        try:
            # 处理特定股票数据
            results = process_specific_stocks(
                tickers=[ticker],
                raw_data_dir=self.raw_data_dir,
                processed_data_dir=self.processed_data_dir,
                test_size=0.0,  # 不需要分割数据
                val_size=0.0,   # 不需要分割数据
                sequence_length=self.sequence_length,
                prediction_horizon=self.prediction_horizon,
                feature_groups=self.feature_groups,
                scaler_type=self.scaler_type
            )
            
            if ticker not in results:
                self.logger.error(f"处理股票 {ticker} 数据失败")
                return None
                
            # 获取处理后的序列
            sequences = results[ticker]['sequences']
            
            # 确保维度一致
            feature_dim = sequences['train'][0].shape[-1]
            if feature_dim != 64:
                self.logger.warning(f"特征维度 {feature_dim} 与预期的 64 不一致，可能影响模型性能")
            
            # 保存训练数据到磁盘以便稍后加载
            X_train = sequences['train'][0]
            y_train = sequences['train'][1]
            
            X_path = os.path.join(self.processed_data_dir, f"{ticker}_X_train.npy")
            y_path = os.path.join(self.processed_data_dir, f"{ticker}_y_train.npy")
            
            np.save(X_path, X_train)
            np.save(y_path, y_train)
            
            self.logger.info(f"已保存处理后的数据到 {X_path}")
            
            # 返回最新序列的特征
            return X_train[-1:] 
        except Exception as e:
            self.logger.error(f"处理数据时出错: {str(e)}")
            return None
    
    def predict(self, ticker: str, inference_times: int = 10) -> Dict:
        """
        对单个股票进行预测
        
        Args:
            ticker (str): 股票代码
            inference_times (int): 推理次数，默认为10
            
        Returns:
            dict: 包含预测结果的字典
        """
        try:
            # 下载最新数据
            self.logger.info(f"下载 {ticker} 最新数据...")
            self._download_latest_data(ticker)
            
            # 处理数据
            self.logger.info(f"处理 {ticker} 数据...")
            features = self._process_data(ticker)
            if features is None:
                return {"ticker": ticker, "error": "处理数据失败"}
            
            # 加载处理后的数据
            self.logger.info(f"加载 {ticker} 处理后的数据...")
            X_path = os.path.join(self.processed_data_dir, f"{ticker}_X_train.npy")
            if not os.path.exists(X_path):
                return {"ticker": ticker, "error": "处理后的数据不存在"}
            
            X = np.load(X_path)
            if X.shape[0] == 0:
                return {"ticker": ticker, "error": "处理后的数据为空"}
            
            # 获取当前时间戳
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 多次推理并计算平均值和方差
            predictions = []
            
            for i in range(inference_times):
                # 对数据添加一个逐渐减少的噪声
                noise_level = 0.01 * (inference_times - i) / inference_times
                X_noisy = X.copy()
                X_noisy += np.random.normal(0, noise_level, X_noisy.shape)
                
                # 获取最后一个样本的索引
                last_sample_idx = X_noisy.shape[0] - 1
                
                # 转换为张量并移动到设备
                # 分批处理以避免超过最大批量大小限制
                batch_size = 32  # 最大批量大小
                all_outputs = []
                
                # 处理完整批次
                for start_idx in range(0, X_noisy.shape[0], batch_size):
                    end_idx = min(start_idx + batch_size, X_noisy.shape[0])
                    batch = X_noisy[start_idx:end_idx]
                    batch_tensor = torch.tensor(batch, dtype=torch.float32).to(self.device)
                    
                    with torch.no_grad():
                        batch_outputs = self.predictor.predict(batch_tensor)
                        all_outputs.append(batch_outputs.cpu().numpy())
                
                # 合并所有批次的输出
                outputs = np.concatenate(all_outputs)
                
                # 获取最后一天的预测结果（最后一个样本的预测值）
                prediction = outputs[last_sample_idx] * 100  # 转换为百分比
                predictions.append(prediction)
            
            # 计算平均值和方差
            mean_prediction = np.mean(predictions)
            variance = np.var(predictions)
            
            # 返回结果
            return {
                "ticker": ticker,
                "mean_prediction": mean_prediction,
                "variance": variance,
                "predictions": predictions,
                "timestamp": timestamp
            }
            
        except Exception as e:
            self.logger.error(f"预测 {ticker} 时出错: {str(e)}")
            return {"ticker": ticker, "error": str(e)}

    def predict_batch(self, tickers: List[str], inference_times: int = 10) -> List[Dict]:
        """
        批量预测多个股票
        
        Args:
            tickers (List[str]): 股票代码列表
            inference_times (int): 推理次数，默认为10
            
        Returns:
            dict: 股票代码到预测结果的映射
        """
        results = {}
        for ticker in tickers:
            self.logger.info(f"预测股票: {ticker}")
            result = self.predict(ticker, inference_times)
            results[ticker] = result
        return results

    def display_results(self, results, batch_mode=False):
        """显示预测结果"""
        if batch_mode:
            print("\n" + "="*50)
            print("🚀 批量股票预测结果")
            print("="*50)
            
            # 检查是否所有预测都失败
            all_failed = True
            result_with_data = None
            
            for ticker, result in results.items():
                if 'error' in result:
                    self.logger.error(f"预测 {ticker} 时出错: {result['error']}")
                    print(f"\n❌ {ticker}: 预测失败 - {result['error']}")
                    continue
                else:
                    all_failed = False
                    result_with_data = result
                
                # 预测值 - 限制预测值在合理范围内
                prediction = result['mean_prediction']
                
                # 检查预测值是否在合理范围内，如果超出则警告并限制
                if abs(prediction) > 10:
                    original_prediction = prediction
                    prediction = max(min(prediction, 10), -10)  # 限制在-10%到10%之间
                    self.logger.warning(f"{ticker} 原始预测值 {original_prediction:.4f}% 超出合理范围，已限制为 {prediction:.4f}%")
                
                # 获取情感方向
                sentiment = "看涨 📈" if prediction > 0 else "看跌 📉"
                # 计算信心水平
                abs_pred = abs(prediction)
                if abs_pred < 0.5:
                    confidence = "低"
                elif abs_pred < 1.5:
                    confidence = "中"
                else:
                    confidence = "高"
                
                # 计算方向一致性
                total_predictions = len(result['predictions'])
                positive_count = sum(1 for p in result['predictions'] if p > 0)
                negative_count = total_predictions - positive_count
                
                if positive_count > negative_count:
                    consensus = f"偏多 ({positive_count}/{total_predictions})"
                elif negative_count > positive_count:
                    consensus = f"偏空 ({negative_count}/{total_predictions})"
                else:
                    consensus = "中性 (50/50)"
                
                # 计算标准差
                std_dev = math.sqrt(result['variance']) if result['variance'] > 0 else 0
                
                # 计算95%置信区间 (同样限制在合理范围内)
                lower_bound = max(prediction - 1.96 * std_dev, -10)
                upper_bound = min(prediction + 1.96 * std_dev, 10)
                
                # 预测区间
                interval = f"[{lower_bound:.4f}%, {upper_bound:.4f}%]"
                
                # 显示结果
                print(f"\n📊 {ticker} | {result['timestamp']}")
                print(f"{'预测方向:':<12} {sentiment}")
                print(f"{'预测值:':<12} {prediction:.4f}%")
                print(f"{'信心水平:':<12} {confidence}")
                print(f"{'预测区间:':<12} {interval}")
                print(f"{'波动率:':<12} {std_dev:.6f}")
                print(f"{'方向一致性:':<12} {consensus}")
                
                # 添加分隔线
                print("-"*40)
            
            # 添加注脚
            if not all_failed and result_with_data:
                print(f"\n注: 预测基于{len(result_with_data['predictions'])}次推理运行，数据更新时间：{result_with_data['timestamp']}")
            elif all_failed:
                print("\n❌ 所有股票预测均失败，请检查数据和日志以排查问题。")
        else:
            # 单个股票预测结果展示
            if 'error' in results:
                self.logger.error(f"预测失败: {results['error']}")
                print(f"\n❌ 预测失败 - {results['error']}")
                return
                
            # 预测值 - 限制预测值在合理范围内
            prediction = results['mean_prediction']
            
            # 检查预测值是否在合理范围内，如果超出则警告并限制
            if abs(prediction) > 10:
                original_prediction = prediction
                prediction = max(min(prediction, 10), -10)  # 限制在-10%到10%之间
                self.logger.warning(f"{results['ticker']} 原始预测值 {original_prediction:.4f}% 超出合理范围，已限制为 {prediction:.4f}%")
            
            # 获取情感方向
            sentiment = "看涨 📈" if prediction > 0 else "看跌 📉"
            # 计算信心水平
            abs_pred = abs(prediction)
            if abs_pred < 0.5:
                confidence = "低"
            elif abs_pred < 1.5:
                confidence = "中"
            else:
                confidence = "高"
                
            # 计算方向一致性
            total_predictions = len(results['predictions'])
            positive_count = sum(1 for p in results['predictions'] if p > 0)
            negative_count = total_predictions - positive_count
            
            if positive_count > negative_count:
                consensus = f"偏多 ({positive_count}/{total_predictions})"
            elif negative_count > positive_count:
                consensus = f"偏空 ({negative_count}/{total_predictions})"
            else:
                consensus = "中性 (50/50)"
                
            # 计算标准差
            std_dev = math.sqrt(results['variance']) if results['variance'] > 0 else 0
            
            # 计算95%置信区间 (同样限制在合理范围内)
            lower_bound = max(prediction - 1.96 * std_dev, -10)
            upper_bound = min(prediction + 1.96 * std_dev, 10)
            
            # 预测区间
            interval = f"[{lower_bound:.4f}%, {upper_bound:.4f}%]"
            
            # 显示结果
            print("\n" + "="*50)
            print(f"🚀 {results['ticker']} 预测结果 | {results['timestamp']}")
            print("="*50)
            print(f"{'预测方向:':<12} {sentiment}")
            print(f"{'预测值:':<12} {prediction:.4f}%")
            print(f"{'信心水平:':<12} {confidence}")
            print(f"{'预测区间:':<12} {interval}")
            print(f"{'波动率:':<12} {std_dev:.6f}")
            print(f"{'方向一致性:':<12} {consensus}")
            
            # 添加注脚
            print(f"\n注: 预测基于{len(results['predictions'])}次推理运行")

def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description="股票价格预测")
        parser.add_argument("--ticker", type=str, default=None, help="股票代码")
        parser.add_argument("--batch", action="store_true", help="批量预测模式")
        parser.add_argument("--inference-times", type=int, default=10, help="推理次数")
        args = parser.parse_args()

    # 设置日志格式
    local_logger = logging.getLogger()
    local_logger.setLevel(logging.INFO)
    if not local_logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        ch.setFormatter(formatter)
        local_logger.addHandler(ch)
    
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    local_logger.info(f"使用设备: {device}")
    
    # 创建预测器实例
    model_path = "models/SpaceExploreAI_best.pt"
    local_logger.info(f"加载模型中: {model_path}")
    
    inferencer = StockPredictor(
        model_path=model_path,
        device=device
    )
    
    # 单只股票预测
    if args.ticker:
        ticker = args.ticker
        local_logger.info(f"预测单只股票: {ticker}")
        result = inferencer.predict(ticker, inference_times=args.inference_times)
        inferencer.display_results(result)
    # 批量预测
    elif args.batch:
        # 默认的批量预测股票列表
        tickers = ['QQQ', 'SPY', 'AAPL', 'NVDA', 'TSLA', 'MSFT', 'AMZN', 'GOOG']
        local_logger.info(f"批量预测股票: {', '.join(tickers)}")
        results = inferencer.predict_batch(tickers, inference_times=args.inference_times)
        inferencer.display_results(results, batch_mode=True)
    else:
        local_logger.info("请指定股票代码或使用--batch参数进行批量预测")

if __name__ == "__main__":
    main() 