import os
import argparse
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Tuple

from ..model.transformer import StockPricePredictor
from ..data.data_processor import StockDataProcessor
from ..train.trainer import create_stock_predictor_from_checkpoint

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_data(ticker: str, processed_data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载测试数据
    
    参数:
        ticker: 股票代码
        processed_data_dir: 处理后数据目录
        
    返回:
        (特征, 目标) 元组
    """
    # 构建测试数据文件路径
    test_file = os.path.join(processed_data_dir, ticker, "test_sequences.npz")
    
    # 检查文件是否存在
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"测试数据文件不存在: {test_file}")
        
    # 加载数据
    test_data = np.load(test_file)
    X_test = test_data["X"]
    y_test = test_data["y"]
    
    return X_test, y_test


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray, prediction_type: str) -> Dict[str, float]:
    """
    计算评估指标
    
    参数:
        predictions: 预测值
        targets: 目标值
        prediction_type: 预测类型，'regression'或'classification'
        
    返回:
        指标字典
    """
    metrics = {}
    
    if prediction_type == "regression":
        # 回归指标
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        # 方向准确率: 预测涨跌方向与实际相符的比例
        pred_direction = np.sign(predictions)
        true_direction = np.sign(targets)
        direction_accuracy = np.mean(pred_direction == true_direction)
        
        metrics.update({
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "direction_accuracy": direction_accuracy
        })
    else:
        # 分类指标
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = targets.astype(np.int64)
        
        accuracy = np.mean(predicted_classes == true_classes)
        
        # 各类别准确率
        class_accuracy = {}
        for c in range(predictions.shape[1]):
            mask = (true_classes == c)
            if np.sum(mask) > 0:
                class_accuracy[f"class_{c}_accuracy"] = np.mean(
                    predicted_classes[mask] == true_classes[mask]
                )
        
        metrics.update({
            "accuracy": accuracy,
            **class_accuracy
        })
            
    return metrics


def plot_predictions(
    predictions: np.ndarray, 
    targets: np.ndarray, 
    prediction_type: str,
    ticker: str,
    output_dir: str
):
    """
    绘制预测结果
    
    参数:
        predictions: 预测值
        targets: 目标值
        prediction_type: 预测类型，'regression'或'classification'
        ticker: 股票代码
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    if prediction_type == "regression":
        # 绘制回归结果
        plt.plot(targets, label='实际值', color='blue')
        plt.plot(predictions, label='预测值', color='red')
        plt.title(f'{ticker} 股价预测结果')
        plt.xlabel('时间步')
        plt.ylabel('归一化收益率')
        plt.legend()
        plt.grid(True)
        
        # 保存图像
        plt.savefig(os.path.join(output_dir, f"{ticker}_regression_predictions.png"))
        
        # 绘制散点图
        plt.figure(figsize=(8, 8))
        plt.scatter(targets, predictions, alpha=0.5)
        plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--', linewidth=2)
        plt.title(f'{ticker} 实际值 vs 预测值')
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.grid(True)
        
        # 保存图像
        plt.savefig(os.path.join(output_dir, f"{ticker}_regression_scatter.png"))
    else:
        # 绘制分类结果
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = targets.astype(np.int64)
        
        # 混淆矩阵
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{ticker} 预测混淆矩阵')
        plt.ylabel('实际类别')
        plt.xlabel('预测类别')
        
        # 保存图像
        plt.savefig(os.path.join(output_dir, f"{ticker}_classification_confusion.png"))
    
    plt.close('all')


def evaluate_model(args):
    """
    评估模型
    
    参数:
        args: 命令行参数
    """
    logger.info(f"加载模型: {args.model_path}")
    
    # 加载模型
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    predictor = create_stock_predictor_from_checkpoint(args.model_path, device)
    
    logger.info(f"预测类型: {predictor.prediction_type}")
    
    # 处理每只股票
    tickers = args.test_data.split(',')
    
    for ticker in tickers:
        logger.info(f"评估股票: {ticker}")
        
        try:
            # 加载测试数据
            X_test, y_test = load_test_data(ticker, args.processed_data_dir)
            logger.info(f"测试数据大小: {X_test.shape}")
            
            # 转换为张量
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            
            # 批量预测
            batch_size = args.batch_size
            all_predictions = []
            
            logger.info("开始预测...")
            for i in tqdm(range(0, len(X_test), batch_size)):
                batch_X = X_test_tensor[i:i+batch_size]
                with torch.no_grad():
                    batch_pred = predictor.predict(batch_X)
                all_predictions.append(batch_pred.cpu().numpy())
            
            # 合并预测结果
            predictions = np.concatenate(all_predictions, axis=0)
            
            # 计算指标
            metrics = calculate_metrics(predictions, y_test, predictor.prediction_type)
            
            # 打印指标
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            logger.info(f"{ticker} 评估指标: {metrics_str}")
            
            # 保存指标到CSV
            metrics_df = pd.DataFrame([metrics])
            metrics_df.insert(0, 'ticker', ticker)
            metrics_file = os.path.join(args.output_dir, f"{ticker}_metrics.csv")
            metrics_df.to_csv(metrics_file, index=False)
            logger.info(f"指标已保存到: {metrics_file}")
            
            # 绘制预测结果
            plot_predictions(
                predictions,
                y_test,
                predictor.prediction_type,
                ticker,
                args.output_dir
            )
            logger.info(f"预测图已保存到: {args.output_dir}")
            
        except Exception as e:
            logger.error(f"评估股票 {ticker} 时出错: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估股票预测模型")
    
    parser.add_argument("--model_path", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--test_data", type=str, required=True, help="测试数据股票代码，多个用逗号分隔")
    parser.add_argument("--processed_data_dir", type=str, default="../data/processed", help="处理后数据目录")
    parser.add_argument("--output_dir", type=str, default="./results", help="结果输出目录")
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    
    args = parser.parse_args()
    
    # 评估模型
    evaluate_model(args) 