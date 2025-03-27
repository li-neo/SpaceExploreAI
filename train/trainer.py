import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from log.logger import get_logger
# 导入MSCNN和T2T模块
from llmps_model.models.mscnn import MSCNN
from llmps_model.models.t2t import T2TExtractor

from model.transformer import StockTransformerModel, StockPricePredictor

# 设置日志
logger = get_logger(__file__, log_file="train_model.log")


class StockModelTrainer:
    """股票预测模型训练器"""
    
    def __init__(
        self,
        model: StockTransformerModel,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        loss_fn_str: str = "cross_entropy",
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
            "log_interval": 1,
            "mixed_precision": True,
            # 学习率调度器配置
            "scheduler_factor": 0.5,        # 学习率衰减因子
            "scheduler_patience": 1,        # 调度器耐心值
            "scheduler_threshold": 0.1,    # 改进阈值
            "scheduler_cooldown": 0,        # 冷却期
            "scheduler_min_lr": 1e-6,       # 最小学习率
            "scheduler_eps": 1e-8,          # 精度
            "scheduler_verbose": True,      # 是否输出日志
            
            # 动态学习率调整配置
            "use_dynamic_lr": False,        # 是否使用动态学习率调整
            "trend_window_size": 3,         # 趋势窗口大小
            "lr_boost_factor": 2.0,         # 学习率临时提升因子
            "stagnation_threshold": 0.01,   # 损失停滞检测阈值
            
            # 周期性学习率调整
            "use_cyclic_lr": False,         # 是否使用周期性学习率
            "cyclic_lr_base_size": 5,       # 周期基础大小（轮次）
            "cyclic_lr_max_factor": 10.0,   # 周期最大学习率因子
            
            # 批次级学习率调整
            "batch_lr_update": False,       # 是否在批次级别调整学习率
            "batch_lr_update_steps": 100,   # 每多少批次调整一次学习率
            "batch_lr_gamma": 0.995,        # 批次级学习率衰减因子
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
            logger.info(f"使用提供的优化器 (当前学习率: {optimizer.param_groups[0]['lr']})")
            self.optimizer = optimizer
            
        # 验证优化器是否正确设置
        if not isinstance(self.optimizer, torch.optim.Optimizer):
            raise TypeError(f"优化器类型错误: {type(self.optimizer)}")
            
        
        if scheduler is None:
            logger.info(f"创建学习率调度器 ReduceLROnPlateau (lr={self.config['learning_rate']}, factor={self.config['scheduler_factor']}, patience={self.config['scheduler_patience']}, threshold={self.config['scheduler_threshold']})")
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config['scheduler_factor'],
                patience=self.config['scheduler_patience'],
                threshold=self.config['scheduler_threshold'],
                threshold_mode='rel',
                cooldown=self.config['scheduler_cooldown'],
                min_lr=self.config['scheduler_min_lr'],
                eps=self.config['scheduler_eps'],
                verbose=self.config['scheduler_verbose']
            )
        else:
            self.scheduler = scheduler
            
        # 设置损失函数  TODO：regression回归任务要设置为MSELoss

        if model.prediction_type == "regression":
            logger.info("使用MSE损失函数")
            self.loss_fn = nn.MSELoss()
        else:
            if loss_fn_str == "MSELoss":
                logger.info("使用MSE损失函数")
                self.loss_fn = nn.MSELoss()
            else:
                logger.info("使用交叉熵损失函数")
                self.loss_fn = nn.CrossEntropyLoss()
            
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
        
        # 批次计数器（用于批次级学习率调整）
        self.global_batch_count = 0
        
    def train_epoch(self) -> float:
        """训练一个轮次，返回平均损失"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        # 增加配置选项用于控制单批次强化训练
        intensive_training = self.config.get("intensive_training", False)
        intensive_iterations = self.config.get("intensive_iterations", 60)
        # 添加批次级验证频率参数
        validate_every_n_batches = self.config.get("validate_every_n_batches", 0)
        # 批次级学习率更新
        batch_lr_update = self.config.get("batch_lr_update", False)
        batch_lr_update_steps = self.config.get("batch_lr_update_steps", 100)
        batch_lr_gamma = self.config.get("batch_lr_gamma", 0.995)
        
        # 初始化LLM-PS的MSCNN和T2T模块
        use_llmps = self.config.get("use_llmps", True)
        if use_llmps:
            try:              
                # 获取序列长度和特征维度
                # sample_batch = next(iter(self.train_dataloader))
                # seq_len = sample_batch[0].shape[2]
                # feature_dim = sample_batch[0].shape[1]
                seq_len = self.config.get("seq_len", 128)
                feature_dim = self.config.get("feature_dim", 64)
                # 保存特征维度为类成员变量
                self.feature_dim = feature_dim
                
                # 初始化MSCNN模块，用于提取时间序列模式
                mscnn_config = {
                    'in_channels': feature_dim,
                    'base_channels': self.config.get("mscnn_base_channels", 64),
                    'ms_blocks': self.config.get("mscnn_ms_blocks", 1),
                    'seq_len': seq_len,
                    'output_dim': self.config.get("mscnn_output_dim", 64)
                }
                self.mscnn = MSCNN(**mscnn_config).to(self.device)
                
                # 初始化T2T模块，用于提取语义特征
                t2t_config = {
                    'in_channels': feature_dim,
                    'patch_size': min(self.config.get("t2t_patch_size", 24), seq_len // 4),
                    'overlap': self.config.get("t2t_overlap", 8),
                    'embed_dim': self.config.get("t2t_embed_dim", 96),
                    'num_encoder_layers': self.config.get("t2t_num_encoder_layers", 4),
                    'num_decoder_layers': self.config.get("t2t_num_decoder_layers", 1),
                    'nhead': self.config.get("t2t_nhead", 4),
                    'dim_feedforward': self.config.get("t2t_dim_feedforward", 384),
                    'dropout': self.config.get("t2t_dropout", 0.1),
                    'mask_ratio': self.config.get("t2t_mask_ratio", 0.75),
                    'vocab_size': 1024,
                    'output_dim': self.config.get("t2t_output_dim", 512)
                }
                self.t2t = T2TExtractor(**t2t_config).to(self.device)
                logger.info(f"已初始化LLM-PS模块：MSCNN和T2T，序列长度={seq_len}，特征维度={feature_dim}")
            except ImportError as e:
                logger.warning(f"导入LLM-PS模块失败，将使用原始输入: {e}")
                use_llmps = False
            except Exception as e:
                logger.error(f"初始化LLM-PS模块时出错: {e}")
                use_llmps = False
                
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        for i, batch in enumerate(progress_bar):
            # 更新全局批次计数器
            self.global_batch_count += 1
            
            # 批次数据移动到设备
            inputs = batch[0].to(self.device)
            targets = batch[1].to(self.device)
            
            # 使用LLM-PS进行数据预处理
            if use_llmps:
                try:
                    # 保存原始输入，用于计算λ约束损失
                    original_inputs = inputs.clone()
                    
                    # 转换输入形状以适应MSCNN [batch_size, seq_len, features] -> [batch_size, features, seq_len]
                    if inputs.dim() == 3 and inputs.shape[1] != self.feature_dim:
                        inputs_mscnn = inputs.transpose(1, 2)
                    else:
                        inputs_mscnn = inputs
                    
                    # 1. 通过MSCNN提取时间序列模式
                    # inputs_mscnn:[batch_size, features, seq_len]
                    # 输出: temporal_features: [batch_size, features, seq_len][16,32,64]
                    temporal_features = self.mscnn(inputs_mscnn)
                    
                    # 2. 通过T2T提取语义特征
                    # inputs_mscnn:[batch_size, features, seq_len]
                    # 输出: semantic_features: [batch_size, seq_len, features][16,32,64]
                    semantic_features = self.t2t(inputs_mscnn, extract_semantics=True)
                    
                    # 确保特征形状为 [batch_size, seq_len, features]
                    # if temporal_features.dim() == 3:  # 当前是 [batch_size, features, seq_len]
                    #     temporal_features = temporal_features.transpose(1, 2)
                    
                    # 3. 合并特征
                    # 现在T2T已经直接返回正确格式 [batch_size, seq_len, features]，不需要再处理tuple和转置
                    
                    # 用增强的特征替换原始输入
                    llmps_inputs = temporal_features + semantic_features
                    
                    # 计算λ约束损失（论文公式(9)）
                    # Lλ = MSE(inputs, projected_features)
                    lambda_loss = nn.MSELoss()(original_inputs, llmps_inputs)
                    lambda_weight = self.config.get("llmps_lambda_weight", 0.01)  # λ值，根据配置设置
                except Exception as e:
                    logger.error(f"LLM-PS预处理失败: {e}")
                    # 失败时使用原始输入
                    inputs = original_inputs
                    lambda_loss = 0
                    lambda_weight = 0
            
            if intensive_training:
                # 对同一批次数据进行多次训练，强化学习
                batch_losses = []
                for j in range(intensive_iterations):
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    predictions = outputs["prediction"]
                    loss = self.loss_fn(predictions, targets)
                    
                    # 添加LLM-PS的约束损失
                    if use_llmps and lambda_weight > 0:
                        loss += lambda_weight * lambda_loss
                        
                    loss.backward()
                    
                    # 梯度裁剪
                    if self.config["clip_grad_norm"] > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config["clip_grad_norm"]
                        )
                    
                    self.optimizer.step()
                    batch_losses.append(loss.item())
                    
                    # 批次级学习率调度器更新
                    if self.config["scheduler_type"] == "batch":
                        self.scheduler.step()
                    
                    # 每隔一定迭代输出一次损失信息
                    if (j + 1) % self.config.get("intensive_log_interval", 10) == 0:
                        lr = self.optimizer.param_groups[0]['lr']
                        logger.info(
                            f"Batch {i+1}/{num_batches}, Iteration {j+1}/{intensive_iterations}, "
                            f"Loss: {loss.item():.4f}, LR: {lr:.6f}"
                        )
                
                # 使用最后一次迭代的损失作为批次损失
                batch_loss = batch_losses[-1]
                # 记录损失变化情况
                if len(batch_losses) > 1:
                    logger.info(
                        f"Batch {i+1}/{num_batches} intensive training complete. "
                        f"Initial loss: {batch_losses[0]:.4f}, Final loss: {batch_losses[-1]:.4f}, "
                        f"Improvement: {batch_losses[0] - batch_losses[-1]:.4f}"
                    )
            else:
                # 正常训练流程
                self.optimizer.zero_grad()
                
                # 混合精度前向传播
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        predictions = outputs["prediction"]
                        loss = self.loss_fn(predictions, targets)
                        
                        # 添加LLM-PS的约束损失
                        if use_llmps and lambda_weight > 0:
                            loss += lambda_weight * lambda_loss
                else:
                    outputs = self.model(inputs)
                    predictions = outputs["prediction"]
                    loss = self.loss_fn(predictions, targets)
                    
                    # 添加LLM-PS的约束损失
                    if use_llmps and lambda_weight > 0:
                        loss += lambda_weight * lambda_loss
                
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
                    try:
                        loss.backward()
                    except Exception as e:
                        logger.error(f"反向传播错误: {e}")
                        raise e
                    
                    # 梯度裁剪
                    if self.config["clip_grad_norm"] > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config["clip_grad_norm"]
                        )
                    
                    # 优化器步进
                    self.optimizer.step()
                
                # 批次级学习率调度器更新
                if self.config["scheduler_type"] == "batch":
                    self.scheduler.step()
            
            # 批次级手动学习率调整
            if batch_lr_update and self.global_batch_count % batch_lr_update_steps == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                new_lr = current_lr * batch_lr_gamma
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                logger.info(f"批次级学习率调整: {current_lr:.6f} -> {new_lr:.6f} (步数: {self.global_batch_count})")
            
            # 立即释放计算图，确保不会重复使用
            if not intensive_training:  # 强化训练模式下不需要手动释放
                del loss, outputs, predictions
            
            # 更新显存（仅在CUDA可用时）
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 更新进度条和累计损失
            epoch_loss += batch_loss
            progress_bar.set_postfix({"loss": f"{batch_loss:.4f}", "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}"})
            
            # 记录日志（非强化训练模式）
            if not intensive_training and (i + 1) % self.config["log_interval"] == 0:
                lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    f"Batch {i+1}/{num_batches}, Loss: {batch_loss:.4f}, LR: {lr:.6f}"
                )
            
            # 在批次训练后执行验证，帮助尽早发现过拟合
            if validate_every_n_batches > 0 and (i + 1) % validate_every_n_batches == 0:
                logger.info(f"Performing validation after batch {i+1}/{num_batches}")
                val_loss, batch_metrics = self.validate()
                # 记录验证指标
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in batch_metrics.items()])
                logger.info(
                    f"Batch validation result: Loss: {val_loss:.4f}, Metrics: {metrics_str}"
                )
                # 恢复训练模式
                self.model.train()
            
            # 在每个批次后清理输入和目标张量以减少内存使用
            del inputs, targets
            if use_llmps and 'original_inputs' in locals():
                del original_inputs
                if 'temporal_features' in locals():
                    del temporal_features
                if 'semantic_features' in locals():
                    del semantic_features
                if 'projected_features' in locals():
                    del projected_features
                
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
        
        # 检查是否使用LLM-PS
        use_llmps = self.config.get("use_llmps", True) and hasattr(self, 'mscnn') and hasattr(self, 't2t')
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                # 批次数据移动到设备
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                
                # 使用LLM-PS进行数据预处理
                if use_llmps:
                    try:
                        # 转换输入形状以适应MSCNN [batch_size, seq_len, features] -> [batch_size, features, seq_len]
                        if inputs.dim() == 3 and inputs.shape[1] != self.feature_dim:
                            inputs_mscnn = inputs.transpose(1, 2)
                        else:
                            inputs_mscnn = inputs
                        
                        # 1. 通过MSCNN提取时间序列模式
                        temporal_features = self.mscnn(inputs_mscnn)
                        # 确保特征形状为 [batch_size, seq_len, features]
                        # if temporal_features.dim() == 3:
                        #      # 当前是 [batch_size, features, seq_len]
                        #     temporal_features = temporal_features.transpose(1, 2)
                        
                        # 2. 通过T2T提取语义特征
                        semantic_features = self.t2t(inputs_mscnn, extract_semantics=True)
                         
                        # 3. 合并特征
                        # 现在T2T已经直接返回正确格式 [batch_size, seq_len, features]，不需要再处理tuple和转置
                        
                        inputs = temporal_features + semantic_features
                                
                        logger.info(f"验证: 原始输入形状: {inputs.shape}, 投影后特征形状: {inputs.shape}")
                    except Exception as e:
                        logger.error(f"验证中LLM-PS预处理失败: {e}")
                
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
        # 早停最小改进阈值
        min_improvement = self.config.get("early_stopping_min_improvement", 0.0)
        
        # 用于动态学习率调整的记录
        val_loss_history = []
        # 是否启用动态学习率调整
        use_dynamic_lr = self.config.get("use_dynamic_lr", False)
        # 趋势窗口大小
        trend_window_size = self.config.get("trend_window_size", 3)
        # 学习率额外乘数
        lr_boost_factor = self.config.get("lr_boost_factor", 2.0)
        # 损失停滞检测阈值
        stagnation_threshold = self.config.get("stagnation_threshold", 0.01)
        
        # 训练循环
        for epoch in range(num_epochs):
            logger.info(f"轮次 {epoch+1}/{num_epochs} 开始")
            
            # 训练一轮
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss, metrics = self.validate()
            self.val_losses.append(val_loss)
            val_loss_history.append(val_loss)
            
            # 记录当前学习率（更新前）
            pre_update_lr = self.optimizer.param_groups[0]['lr']
            
            # 动态学习率调整：根据验证损失趋势调整学习率
            if use_dynamic_lr and epoch >= trend_window_size:
                # 计算近期验证损失趋势
                recent_losses = val_loss_history[-trend_window_size:]
                loss_diff = recent_losses[0] - recent_losses[-1]
                loss_rate = loss_diff / recent_losses[0] if recent_losses[0] != 0 else 0
                
                # 检测损失停滞情况
                is_stagnating = abs(loss_rate) < stagnation_threshold
                # 将原来的条件修改得更严格，至少需要明显的连续增长
                recent_losses = val_loss_history[-trend_window_size:]
                avg_loss_increase = sum(recent_losses[i+1] - recent_losses[i] for i in range(len(recent_losses)-1)) / (len(recent_losses)-1)
  
                # 只有当平均增长率大于某个阈值时才降低学习率
                if all(recent_losses[i] < recent_losses[i+1] for i in range(len(recent_losses)-1)) and avg_loss_increase > 0.005:
                    # 损失连续增加且平均增长幅度较大
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= self.config["scheduler_factor"] * 0.8
                    logger.info(f"检测到显著连续损失增加，快速降低学习率: {pre_update_lr:.6f} -> {self.optimizer.param_groups[0]['lr']:.6f}")
                # 损失停滞
                elif is_stagnating:
                    # 如果损失变化很小，尝试临时增加学习率以跳出局部最小值
                    if random.random() < 0.5:  # 50%概率进行学习率提升
                        old_lr = self.optimizer.param_groups[0]['lr']
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] *= lr_boost_factor
                        logger.info(f"损失停滞，尝试提升学习率以逃离局部最小值: {old_lr:.6f} -> {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 标准学习率调度器更新
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 获取更新后的学习率并记录
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # 如果学习率发生变化，记录变化情况
            if current_lr != pre_update_lr:
                logger.info(f"学习率已更新: {pre_update_lr:.6f} -> {current_lr:.6f}")
                    
            # 记录指标
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            logger.info(
                f"轮次 {epoch+1}/{num_epochs} 完成: "
                f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, "
                f"指标: {metrics_str}, LR: {current_lr:.6f}"
            )
            
            # 检查是否为最佳模型 (考虑最小改进阈值)
            if best_val_loss - val_loss > min_improvement:
                improvement = best_val_loss - val_loss
                logger.info(f"验证损失改进: {best_val_loss:.4f} -> {val_loss:.4f} (改进: {improvement:.4f})")
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                
                # 保存最佳模型
                self.save_model(f"{self.config['model_name']}_best.pt")
            else:
                patience_counter += 1
                # 如果性能接近最佳值但未超过阈值，给出提示
                if best_val_loss - val_loss > 0:
                    logger.info(
                        f"验证损失轻微改进 {best_val_loss:.4f} -> {val_loss:.4f} "
                        f"(改进: {best_val_loss - val_loss:.4f})，但未超过阈值 {min_improvement:.4f}"
                    )
                else:
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
            logger.warning("No training data available for plotting")
            return
            
        plt.rcParams['font.sans-serif'] = ['Arial']  # 使用Arial字体
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # 绘制损失曲线
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制学习率曲线
        ax2.plot(epochs, self.learning_rates, 'g-')
        ax2.set_title('Learning Rate')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Learning Rate')
        ax2.set_yscale('log')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(self.config["save_dir"], f"{self.config['model_name']}_training_curves.png")
        plt.savefig(save_path)
        logger.info(f"Training curves saved to {save_path}")
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