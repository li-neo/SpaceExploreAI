import os
import argparse
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Tuple

from model.transformer import StockPricePredictor
from data.data_processor import StockDataProcessor
from train.trainer import create_stock_predictor_from_checkpoint

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
    # 检查处理后数据目录是否存在
    if not os.path.exists(processed_data_dir):
        raise FileNotFoundError(f"处理后数据目录不存在: {processed_data_dir}，请先运行数据预处理步骤")
    
    # 构建测试数据文件路径
    test_file = f"{processed_data_dir}/{ticker}_test_sequences.npz"
    
    # 检查文件是否存在
    if not os.path.exists(test_file):
        # 检查目录中有哪些文件可用
        available_files = os.listdir(processed_data_dir)
        raise FileNotFoundError(
            f"测试数据文件不存在: {test_file}\n"
            f"目录 {processed_data_dir} 中的可用文件: {available_files}\n"
            f"请确保已为股票 {ticker} 生成测试数据"
        )
        
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
    output_dir: str,
    metrics: Dict[str, float] = None
):
    """
    绘制预测结果
    
    参数:
        predictions: 预测值
        targets: 目标值
        prediction_type: 预测类型，'regression'或'classification'
        ticker: 股票代码
        output_dir: 输出目录
        metrics: 评估指标字典
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    if prediction_type == "regression":
        # 绘制回归结果
        plt.plot(targets, label='Actual', color='blue')
        plt.plot(predictions, label='Predicted', color='red')
        plt.title(f'{ticker} Stock Price Prediction Results')
        plt.xlabel('Time Steps')
        plt.ylabel('Normalized Returns')
        
        # 添加评估指标信息
        if metrics is not None:
            metrics_text = (f"MSE: {metrics.get('mse', 0):.4f}, "
                           f"RMSE: {metrics.get('rmse', 0):.4f}\n"
                           f"MAE: {metrics.get('mae', 0):.4f}, "
                           f"Direction Acc: {metrics.get('direction_accuracy', 0):.2%}")
            plt.figtext(0.5, 0.01, metrics_text, ha='center', fontsize=10, 
                       bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
        
        plt.legend()
        plt.grid(True)
        
        # 保存图像
        plt.savefig(os.path.join(output_dir, f"{ticker}_regression_predictions.png"), 
                   bbox_inches='tight', dpi=120)
        
        # 绘制散点图
        plt.figure(figsize=(8, 8))
        plt.scatter(targets, predictions, alpha=0.5)
        plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--', linewidth=2)
        plt.title(f'{ticker} Actual vs Predicted')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        
        # 添加评估指标信息
        if metrics is not None:
            metrics_text = (f"MSE: {metrics.get('mse', 0):.4f}, "
                           f"RMSE: {metrics.get('rmse', 0):.4f}\n"
                           f"MAE: {metrics.get('mae', 0):.4f}, "
                           f"Direction Acc: {metrics.get('direction_accuracy', 0):.2%}")
            plt.figtext(0.5, 0.01, metrics_text, ha='center', fontsize=10, 
                       bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
        
        plt.grid(True)
        
        # 保存图像
        plt.savefig(os.path.join(output_dir, f"{ticker}_regression_scatter.png"), 
                   bbox_inches='tight', dpi=120)
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
        plt.title(f'{ticker} Prediction Confusion Matrix')
        plt.ylabel('Actual Class')
        plt.xlabel('Predicted Class')
        
        # 添加评估指标信息
        if metrics is not None:
            metrics_text = f"Accuracy: {metrics.get('accuracy', 0):.2%}"
            plt.figtext(0.5, 0.01, metrics_text, ha='center', fontsize=10, 
                       bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
        
        # 保存图像
        plt.savefig(os.path.join(output_dir, f"{ticker}_classification_confusion.png"), 
                   bbox_inches='tight', dpi=120)
    
    plt.close('all')


def evaluate_model(args):
    """
    评估模型
    
    参数:
        args: 命令行参数
    """
    logger.info(f"加载模型: {args.model_path}")
    
    try:
        # 使用正确的设备 - 优先使用MPS (Mac GPU)
        if torch.backends.mps.is_available() and args.device == 'mps':
            device = torch.device('mps')
            logger.info("使用 MPS 设备进行预测")
        elif torch.cuda.is_available() and args.device == 'cuda':
            device = torch.device('cuda')
            logger.info("使用 CUDA 设备进行预测")
        else:
            device = torch.device('cpu')
            logger.info("使用 CPU 设备进行预测")
            
        # 加载模型
        predictor = create_stock_predictor_from_checkpoint(args.model_path, device)
        
        logger.info(f"预测类型: {predictor.prediction_type}")
        logger.info(f"模型数据类型: {next(predictor.model.parameters()).dtype}")
        
        # 处理每只股票
        tickers = args.test_data.split(',')
        
        for ticker in tickers:
            ticker = ticker.strip()  # 移除可能的空格
            logger.info(f"评估股票: {ticker}")
            
            try:
                # 加载测试数据
                X_test, y_test = load_test_data(ticker, args.processed_data_dir)
                logger.info(f"测试数据大小: {X_test.shape}")
                
                # 获取模型的数据类型
                model_dtype = next(predictor.model.parameters()).dtype
                logger.info(f"使用模型数据类型: {model_dtype}")
                
                # 转换为张量 - 使用与模型相同的数据类型
                X_test_tensor = torch.tensor(X_test, dtype=model_dtype)
                
                # 批量预测
                batch_size = args.batch_size
                all_predictions = []
                
                logger.info("开始预测...")
                for i in tqdm(range(0, len(X_test), batch_size)):
                    batch_X = X_test_tensor[i:i+batch_size].to(device)
                    with torch.no_grad():
                        batch_pred = predictor.predict(batch_X)
                    all_predictions.append(batch_pred.cpu().numpy())
                    
                # 合并预测结果
                predictions = np.concatenate(all_predictions, axis=0)
                
                # 检查预测结果是否与目标形状一致
                if predictions.shape[0] != y_test.shape[0]:
                    logger.warning(f"预测结果形状 {predictions.shape} 与目标形状 {y_test.shape} 不一致")
                    # 调整预测结果或目标值，确保形状匹配
                    min_len = min(predictions.shape[0], y_test.shape[0])
                    predictions = predictions[:min_len]
                    y_test = y_test[:min_len]
                
                # 计算指标
                metrics = calculate_metrics(predictions, y_test, predictor.prediction_type)
                
                # 打印指标
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                logger.info(f"{ticker} 评估指标: {metrics_str}")
                
                # 保存指标到CSV
                metrics_df = pd.DataFrame([metrics])
                metrics_df.insert(0, 'ticker', ticker)
                metrics_file = os.path.join(args.output_dir, f"{ticker}_metrics.csv")
                os.makedirs(args.output_dir, exist_ok=True)  # 确保输出目录存在
                metrics_df.to_csv(metrics_file, index=False)
                logger.info(f"指标已保存到: {metrics_file}")
                
                # 绘制预测结果
                plot_predictions(
                    predictions,
                    y_test,
                    predictor.prediction_type,
                    ticker,
                    args.output_dir,
                    metrics
                )
                logger.info(f"预测图已保存到: {args.output_dir}")
                
            except Exception as e:
                logger.error(f"评估股票 {ticker} 时出错: {str(e)}", exc_info=True)
    except Exception as e:
        logger.error(f"模型评估过程中出错: {str(e)}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估股票预测模型")
    
    parser.add_argument("--model_path", default="./models/SpaceExploreAI_best.pt", type=str, help="模型检查点路径")
    parser.add_argument("--test_data", default="TSLA,NVDA,AAPL,GOOG,QQQ", type=str, help="测试数据股票代码，多个用逗号分隔")
    parser.add_argument("--processed_data_dir", type=str, default="./data/processed/test/", help="处理后数据目录")
    parser.add_argument("--output_dir", type=str, default="./results/predict", help="结果输出目录")
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    
    # 检查MPS是否可用，如果可用，默认使用MPS
    default_device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--device", type=str, default=default_device, help="设备 (cpu, cuda, mps)")
    
    args = parser.parse_args()
    
    # 评估模型
    evaluate_model(args) 