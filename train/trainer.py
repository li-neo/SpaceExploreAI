import os
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
import matplotlib.pyplot as plt
from tqdm import tqdm

from model.transformer import StockTransformerModel, StockPricePredictor

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockModelTrainer:
    """股票预测模型训练器"""
    
    def __init__(
        self,
        model: StockTransformerModel,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        loss_fn: Callable = None,
        device: str = None,
        config: Dict = None
    ):
        """
        初始化训练器
        
        参数:
            model: 模型实例
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
            loss_fn: 损失函数
            device: 设备
            config: 配置字典
        """
        # 检查模型是否为None
        if model is None:
            raise ValueError("模型不能为None")
            
        # 检查数据加载器是否为None
        if train_dataloader is None or val_dataloader is None:
            raise ValueError("训练数据加载器和验证数据加载器不能为None")
            
        # 设置设备
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
            
        logger.info(f"使用设备: {self.device}")
            
        # 设置模型
        self.model = model.to(self.device)
        
        # 设置数据加载器
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # 设置配置
        self.config = {
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "clip_grad_norm": 1.0,
            "num_epochs": 30,
            "patience": 5,
            "save_dir": "./models",
            "model_name": "stock_transformer",
            "log_interval": 10,
            "mixed_precision": True
        }
        if config:
            self.config.update(config)
            
        # 创建保存目录
        os.makedirs(self.config["save_dir"], exist_ok=True)
        
        # 设置优化器
        if optimizer is None:
            logger.info(f"创建优化器 AdamW (lr={self.config['learning_rate']}, weight_decay={self.config['weight_decay']})")
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"]
            )
        else:
            logger.info("使用提供的优化器")
            self.optimizer = optimizer
            
        # 验证优化器是否正确设置
        if not isinstance(self.optimizer, torch.optim.Optimizer):
            raise TypeError(f"优化器类型错误: {type(self.optimizer)}")
            
        # 设置学习率调度器
        if scheduler is None:
            logger.info("创建学习率调度器 ReduceLROnPlateau")
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=2,
                verbose=True
            )
        else:
            logger.info("使用提供的学习率调度器")
            self.scheduler = scheduler
            
        # 验证学习率调度器是否正确设置
        if not isinstance(self.scheduler, torch.optim.lr_scheduler._LRScheduler) and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            raise TypeError(f"学习率调度器类型错误: {type(self.scheduler)}")
            
        # 设置损失函数
        if loss_fn is None:
            if model.prediction_type == "regression":
                logger.info("使用MSE损失函数")
                self.loss_fn = nn.MSELoss()
            else:
                logger.info("使用交叉熵损失函数")
                self.loss_fn = nn.CrossEntropyLoss()
        else:
            logger.info("使用提供的损失函数")
            self.loss_fn = loss_fn
            
        # 设置混合精度训练 - MPS和CPU不支持混合精度训练
        self.use_mixed_precision = self.config["mixed_precision"] and torch.cuda.is_available()
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            self.scaler = None
            if self.config["mixed_precision"]:
                logger.warning("混合精度训练仅支持CUDA设备，已自动禁用")
        
        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.epochs_trained = 0
        
    def train_epoch(self) -> float:
        """训练一个轮次，返回平均损失"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        for i, batch in enumerate(progress_bar):
            # 批次数据移动到设备
            inputs = batch[0].to(self.device)
            targets = batch[1].to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 混合精度前向传播
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    predictions = outputs["prediction"]
                    loss = self.loss_fn(predictions, targets)
            else:
                outputs = self.model(inputs)
                predictions = outputs["prediction"]
                loss = self.loss_fn(predictions, targets)
            
            # 保存loss值以供后续使用
            batch_loss = loss.item()
                
            # 反向传播
            if self.scaler:
                # 使用混合精度时的反向传播
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                if self.config["clip_grad_norm"] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config["clip_grad_norm"]
                    )
                    
                # 优化器步进
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 普通反向传播
                loss.backward()
                
                # 梯度裁剪
                if self.config["clip_grad_norm"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config["clip_grad_norm"]
                    )
                
                # 优化器步进
                self.optimizer.step()
            
            # 立即释放计算图，确保不会重复使用
            del loss, outputs, predictions
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # 只在CUDA可用时清理GPU内存
            
            # 更新进度条和累计损失
            epoch_loss += batch_loss
            progress_bar.set_postfix({"loss": f"{batch_loss:.4f}"})
            
            # 记录日志
            if (i + 1) % self.config["log_interval"] == 0:
                lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    f"Batch {i+1}/{num_batches}, Loss: {batch_loss:.4f}, LR: {lr:.6f}"
                )
                
        # 计算平均损失
        avg_loss = epoch_loss / num_batches
        return avg_loss
    
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """验证模型，返回平均损失和指标"""
        self.model.eval()
        val_loss = 0.0
        num_batches = len(self.val_dataloader)
        
        # 指标计算准备
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                # 批次数据移动到设备
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                
                # 前向传播
                outputs = self.model(inputs)
                predictions = outputs["prediction"]
                
                # 计算损失
                loss = self.loss_fn(predictions, targets)
                val_loss += loss.item()
                
                # 收集所有预测和目标用于指标计算
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                
        # 计算平均损失
        avg_loss = val_loss / num_batches
        
        # 计算指标
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        metrics = self.calculate_metrics(all_predictions, all_targets)
        
        return avg_loss, metrics
    
    def calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """计算评估指标"""
        metrics = {}
        
        if self.model.prediction_type == "regression":
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
    
    def train(self, num_epochs: int = None) -> Dict:
        """
        训练模型
        
        参数:
            num_epochs: 训练轮次

        返回:
            训练记录字典
        """
        if num_epochs is None:
            num_epochs = self.config["num_epochs"]
            
        logger.info(f"开始训练，共 {num_epochs} 轮")
        
        # 记录开始时间
        start_time = time.time()
        
        # 最佳验证损失和轮次
        best_val_loss = float('inf')
        best_epoch = -1
        
        # 早停计数器
        patience_counter = 0
        
        # 训练循环
        for epoch in range(num_epochs):
            logger.info(f"轮次 {epoch+1}/{num_epochs} 开始")
            
            # 训练一轮
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss, metrics = self.validate()
            self.val_losses.append(val_loss)
            
            # 记录当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # 更新学习率调度器
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                    
            # 记录指标
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            logger.info(
                f"轮次 {epoch+1}/{num_epochs} 完成: "
                f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, "
                f"指标: {metrics_str}, LR: {current_lr:.6f}"
            )
            
            # 检查是否为最佳模型
            if val_loss < best_val_loss:
                logger.info(f"验证损失改进: {best_val_loss:.4f} -> {val_loss:.4f}")
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                
                # 保存最佳模型
                self.save_model(f"{self.config['model_name']}_best.pt")
            else:
                patience_counter += 1
                logger.info(f"验证损失未改进，耐心计数: {patience_counter}/{self.config['patience']}")
                
            # 保存最新模型
            self.save_model(f"{self.config['model_name']}_last.pt")
            
            # 更新已训练轮数
            self.epochs_trained = epoch + 1
                
            # 检查是否需要早停
            if patience_counter >= self.config["patience"]:
                logger.info(f"早停触发，最佳轮次: {best_epoch+1}，最佳验证损失: {best_val_loss:.4f}")
                break
                
        # 计算总训练时间
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(
            f"训练完成，总时长: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒, "
            f"最佳轮次: {best_epoch+1}, 最佳验证损失: {best_val_loss:.4f}"
        )
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        # 返回训练记录
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "epochs_trained": self.epochs_trained
        }
    
    def save_model(self, path: str):
        """
        保存模型和训练状态
        
        参数:
            path: 保存路径
        """
        # 创建保存字典
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "epochs_trained": self.epochs_trained,
            "vocab_size": self.model.vocab_size,
            "hidden_size": self.model.hidden_size,
            "prediction_type": self.model.prediction_type
        }
        
        # 如果调度器存在且是可序列化的，保存其状态
        if self.scheduler and hasattr(self.scheduler, 'state_dict'):
            save_dict["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # 保存到文件
        save_path = os.path.join(self.config["save_dir"], path)
        torch.save(save_dict, save_path)
        logger.info(f"模型已保存到 {save_path}")
    
    def load_model(self, path: str):
        """
        加载模型和训练状态
        
        参数:
            path: 模型路径
        """
        # 加载检查点
        checkpoint = torch.load(path, map_location=self.device)
        
        # 加载模型状态
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # 加载优化器状态（如果存在）
        if "optimizer_state_dict" in checkpoint and self.optimizer:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                # 将优化器状态移动到正确的设备
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
                logger.info("优化器状态已加载")
            except Exception as e:
                logger.warning(f"加载优化器状态失败: {e}")
        
        # 加载调度器状态（如果存在）
        if "scheduler_state_dict" in checkpoint and self.scheduler and hasattr(self.scheduler, 'load_state_dict'):
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logger.info("学习率调度器状态已加载")
            except Exception as e:
                logger.warning(f"加载学习率调度器状态失败: {e}")
        
        # 加载训练记录
        if "train_losses" in checkpoint:
            self.train_losses = checkpoint["train_losses"]
        if "val_losses" in checkpoint:
            self.val_losses = checkpoint["val_losses"]
        if "learning_rates" in checkpoint:
            self.learning_rates = checkpoint["learning_rates"]
        if "epochs_trained" in checkpoint:
            self.epochs_trained = checkpoint["epochs_trained"]
            
        logger.info(f"模型已从 {path} 加载，之前已训练 {self.epochs_trained} 轮")
        
    def plot_training_curves(self):
        """绘制训练曲线"""
        if not self.train_losses or not self.val_losses:
            logger.warning("没有训练数据可供绘制")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # 绘制损失曲线
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='训练损失')
        ax1.plot(epochs, self.val_losses, 'r-', label='验证损失')
        ax1.set_title('训练和验证损失')
        ax1.set_xlabel('轮次')
        ax1.set_ylabel('损失')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制学习率曲线
        ax2.plot(epochs, self.learning_rates, 'g-')
        ax2.set_title('学习率')
        ax2.set_xlabel('轮次')
        ax2.set_ylabel('学习率')
        ax2.set_yscale('log')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(self.config["save_dir"], f"{self.config['model_name']}_training_curves.png")
        plt.savefig(save_path)
        logger.info(f"训练曲线已保存到 {save_path}")
        plt.close()


def create_stock_predictor_from_checkpoint(
    checkpoint_path: str,
    device: str = None
) -> StockPricePredictor:
    """
    从检查点创建股票预测器
    
    参数:
        checkpoint_path: 检查点路径
        device: 设备
        
    返回:
        股票预测器实例
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 创建预测器
    predictor = StockPricePredictor(
        feature_dim=checkpoint["vocab_size"],
        hidden_size=checkpoint["hidden_size"],
        prediction_type=checkpoint["prediction_type"],
        device=device
    )
    
    # 加载模型权重
    predictor.model.load_state_dict(checkpoint["model_state_dict"])
    
    return predictor 