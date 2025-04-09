import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

from log.logger import get_logger
from RLHF.dpo_args import DPOArgs
from RLHF.data_processor import PreferenceDataProcessor

# 设置日志
logger = get_logger(__file__, log_file="dpo_trainer.log")


class DPOTrainer:
    """
    Direct Preference Optimization (DPO) 训练器
    
    DPO算法基于以下论文实现:
    "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
    (Rafailov et al., 2023)
    """
    
    def __init__(
        self,
        model: nn.Module,
        reference_model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer = None,
        args: DPOArgs = None,
        device: torch.device = None
    ):
        """
        初始化DPO训练器
        
        参数:
            model: 要优化的模型
            reference_model: 参考模型（通常是SFT模型）
            train_dataloader: 训练数据加载器
            eval_dataloader: 评估数据加载器
            optimizer: 优化器
            args: DPO参数
            device: 训练设备
        """
        # 设置模型和参考模型
        self.model = model
        self.reference_model = reference_model
        
        # 冻结参考模型参数
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        # 设置数据加载器
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # 设置参数
        self.args = args if args is not None else DPOArgs()
        
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
        
        # 将模型移动到设备
        self.model = self.model.to(self.device)
        self.reference_model = self.reference_model.to(self.device)
        
        # 设置优化器
        if optimizer is None:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.args.dpo_learning_rate,
                weight_decay=self.args.weight_decay
            )
        else:
            self.optimizer = optimizer
        
        # 创建保存目录
        os.makedirs(self.args.save_dir, exist_ok=True)
        
        # 设置混合精度训练
        self.use_mixed_precision = False
        if self.args.dpo_mixed_precision and torch.cuda.is_available():
            self.use_mixed_precision = True
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        # 跟踪训练指标
        self.train_losses = []
        self.eval_losses = []
        self.learning_rates = []
        self.preference_accuracies = []
    
    def _compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算DPO损失
        
        参数:
            policy_chosen_logps: 策略模型对胜者的对数概率
            policy_rejected_logps: 策略模型对败者的对数概率
            reference_chosen_logps: 参考模型对胜者的对数概率
            reference_rejected_logps: 参考模型对败者的对数概率
            
        返回:
            损失值, 准确率, 奖励差
        """
        # 计算策略和参考策略之间的对数概率差
        policy_chosen_logps = policy_chosen_logps.squeeze(-1)
        policy_rejected_logps = policy_rejected_logps.squeeze(-1)
        reference_chosen_logps = reference_chosen_logps.squeeze(-1)
        reference_rejected_logps = reference_rejected_logps.squeeze(-1)
        
        chosen_rewards = policy_chosen_logps - reference_chosen_logps
        rejected_rewards = policy_rejected_logps - reference_rejected_logps
        
        # 计算奖励差值
        reward_diffs = chosen_rewards - rejected_rewards
        
        if self.args.loss_type == "sigmoid":
            # 使用sigmoid损失（原始DPO损失）
            losses = -F.logsigmoid(self.args.beta * reward_diffs)
        elif self.args.loss_type == "hinge":
            # 使用hinge损失
            losses = torch.relu(1 - self.args.beta * reward_diffs)
        else:
            raise ValueError(f"未知的损失类型: {self.args.loss_type}")
        
        # 计算精度（正确分类偏好对的比例）
        accuracies = (reward_diffs > 0).float().mean()
        
        # 返回损失、精度和奖励差
        return losses.mean(), accuracies, reward_diffs.mean()
    
    def _get_batch_logps(
        self,
        model: nn.Module,
        prompt_tensors: torch.Tensor,
        response_tensors: torch.Tensor
    ) -> torch.Tensor:
        """
        获取给定模型对一批响应的对数概率
        
        参数:
            model: 模型
            prompt_tensors: 输入提示张量
            response_tensors: 响应张量
            
        返回:
            对数概率张量
        """
        # 模型推理
        with torch.no_grad() if model is self.reference_model else torch.enable_grad():
            outputs = model(prompt_tensors)
            
            # 获取模型预测的logits
            outputs = model.compute_loss(outputs, response_tensors, return_logits=True)
            
            # 计算对数概率
            logps = -F.mse_loss(outputs, response_tensors, reduction="none")
            
            # 沿着序列维度取平均
            logps = logps.mean(dim=[-1, -2])
            
            return logps
    
    def train_step(
        self,
        prompts: torch.Tensor,
        chosen_responses: torch.Tensor,
        rejected_responses: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        执行一步DPO训练
        
        参数:
            prompts: 输入提示
            chosen_responses: 胜者响应
            rejected_responses: 败者响应
            
        返回:
            包含训练指标的字典
        """
        # 将输入移到设备
        prompts = prompts.to(self.device)
        chosen_responses = chosen_responses.to(self.device)
        rejected_responses = rejected_responses.to(self.device)
        
        # 启用混合精度训练
        if self.use_mixed_precision:
            with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.args.dpo_mixed_precision == "fp16" else torch.bfloat16):
                # 计算策略模型概率
                policy_chosen_logps = self._get_batch_logps(
                    self.model, prompts, chosen_responses
                )
                policy_rejected_logps = self._get_batch_logps(
                    self.model, prompts, rejected_responses
                )
                
                # 计算参考模型概率
                with torch.no_grad():
                    reference_chosen_logps = self._get_batch_logps(
                        self.reference_model, prompts, chosen_responses
                    )
                    reference_rejected_logps = self._get_batch_logps(
                        self.reference_model, prompts, rejected_responses
                    )
                
                # 计算DPO损失
                loss, accuracy, reward_diff = self._compute_dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    reference_chosen_logps,
                    reference_rejected_logps
                )
            
            # 反向传播
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            if self.args.dpo_max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.dpo_max_grad_norm
                )
            
            # 优化器步进
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # 计算策略模型概率
            policy_chosen_logps = self._get_batch_logps(
                self.model, prompts, chosen_responses
            )
            policy_rejected_logps = self._get_batch_logps(
                self.model, prompts, rejected_responses
            )
            
            # 计算参考模型概率
            with torch.no_grad():
                reference_chosen_logps = self._get_batch_logps(
                    self.reference_model, prompts, chosen_responses
                )
                reference_rejected_logps = self._get_batch_logps(
                    self.reference_model, prompts, rejected_responses
                )
            
            # 计算DPO损失
            loss, accuracy, reward_diff = self._compute_dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.args.dpo_max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.dpo_max_grad_norm
                )
            
            # 优化器步进
            self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "reward_diff": reward_diff.item()
        }
    
    def eval_step(
        self,
        prompts: torch.Tensor,
        chosen_responses: torch.Tensor,
        rejected_responses: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        执行一步DPO评估
        
        参数:
            prompts: 输入提示
            chosen_responses: 胜者响应
            rejected_responses: 败者响应
            
        返回:
            包含评估指标的字典
        """
        # 将输入移到设备
        prompts = prompts.to(self.device)
        chosen_responses = chosen_responses.to(self.device)
        rejected_responses = rejected_responses.to(self.device)
        
        # 设置为评估模式
        self.model.eval()
        
        with torch.no_grad():
            # 计算策略模型概率
            policy_chosen_logps = self._get_batch_logps(
                self.model, prompts, chosen_responses
            )
            policy_rejected_logps = self._get_batch_logps(
                self.model, prompts, rejected_responses
            )
            
            # 计算参考模型概率
            reference_chosen_logps = self._get_batch_logps(
                self.reference_model, prompts, chosen_responses
            )
            reference_rejected_logps = self._get_batch_logps(
                self.reference_model, prompts, rejected_responses
            )
            
            # 计算DPO损失
            loss, accuracy, reward_diff = self._compute_dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps
            )
        
        # 恢复为训练模式
        self.model.train()
        
        return {
            "eval_loss": loss.item(),
            "eval_accuracy": accuracy.item(),
            "eval_reward_diff": reward_diff.item()
        }
    
    def train(self):
        """
        执行DPO训练
        
        返回:
            训练历史
        """
        logger.info("开始DPO训练...")
        
        best_eval_loss = float('inf')
        steps_since_improvement = 0
        global_step = 0
        
        # 训练历史
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "train_reward_diff": [],
            "eval_loss": [],
            "eval_accuracy": [],
            "eval_reward_diff": [],
            "learning_rate": []
        }
        
        # 训练循环
        for epoch in range(self.args.dpo_num_epochs):
            epoch_start_time = time.time()
            total_loss = 0.0
            total_accuracy = 0.0
            total_reward_diff = 0.0
            
            # 创建进度条
            train_iter = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch+1}/{self.args.dpo_num_epochs}",
                leave=False
            )
            
            # 遍历训练数据
            for step, batch in enumerate(train_iter):
                # 获取批次数据
                prompts = batch["prompt"]
                chosen_responses = batch["chosen"]
                rejected_responses = batch["rejected"]
                
                # 训练步骤
                step_results = self.train_step(
                    prompts, chosen_responses, rejected_responses
                )
                
                # 更新总指标
                total_loss += step_results["loss"]
                total_accuracy += step_results["accuracy"]
                total_reward_diff += step_results["reward_diff"]
                
                # 更新进度条
                train_iter.set_postfix({
                    "loss": step_results["loss"],
                    "acc": step_results["accuracy"],
                    "reward_diff": step_results["reward_diff"]
                })
                
                # 执行评估
                if (global_step + 1) % self.args.eval_every == 0:
                    eval_results = self.evaluate()
                    
                    # 更新历史
                    history["eval_loss"].append(eval_results["eval_loss"])
                    history["eval_accuracy"].append(eval_results["eval_accuracy"])
                    history["eval_reward_diff"].append(eval_results["eval_reward_diff"])
                    
                    # 检查改进
                    if eval_results["eval_loss"] < best_eval_loss:
                        best_eval_loss = eval_results["eval_loss"]
                        steps_since_improvement = 0
                        
                        # 保存最佳模型
                        self.save_model(os.path.join(self.args.save_dir, f"{self.args.model_name}_best.pt"))
                    else:
                        steps_since_improvement += 1
                
                global_step += 1
            
            # 计算epoch平均指标
            avg_loss = total_loss / len(self.train_dataloader)
            avg_accuracy = total_accuracy / len(self.train_dataloader)
            avg_reward_diff = total_reward_diff / len(self.train_dataloader)
            
            # 更新历史
            history["train_loss"].append(avg_loss)
            history["train_accuracy"].append(avg_accuracy)
            history["train_reward_diff"].append(avg_reward_diff)
            history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])
            
            # 计算epoch耗时
            epoch_time = time.time() - epoch_start_time
            
            # 打印epoch统计
            logger.info(
                f"Epoch {epoch+1}/{self.args.dpo_num_epochs} - "
                f"Loss: {avg_loss:.6f}, Accuracy: {avg_accuracy:.4f}, "
                f"Reward Diff: {avg_reward_diff:.4f}, Time: {epoch_time:.2f}s"
            )
            
            # 保存检查点
            if (epoch + 1) % 1 == 0:
                self.save_model(os.path.join(self.args.save_dir, f"{self.args.model_name}_epoch{epoch+1}.pt"))
            
            # 早停
            if steps_since_improvement >= self.args.patience:
                logger.info(f"早停触发：{self.args.patience} 轮无改进")
                break
        
        # 保存最终模型
        self.save_model(os.path.join(self.args.save_dir, f"{self.args.model_name}_final.pt"))
        
        logger.info("DPO训练完成！")
        
        return history
    
    def evaluate(self) -> Dict[str, float]:
        """
        在评估集上评估模型
        
        返回:
            包含评估指标的字典
        """
        # 设置为评估模式
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        total_reward_diff = 0.0
        
        # 遍历评估数据
        for batch in self.eval_dataloader:
            # 获取批次数据
            prompts = batch["prompt"]
            chosen_responses = batch["chosen"]
            rejected_responses = batch["rejected"]
            
            # 评估步骤
            step_results = self.eval_step(
                prompts, chosen_responses, rejected_responses
            )
            
            # 更新总指标
            total_loss += step_results["eval_loss"]
            total_accuracy += step_results["eval_accuracy"]
            total_reward_diff += step_results["eval_reward_diff"]
        
        # 计算平均指标
        avg_loss = total_loss / len(self.eval_dataloader)
        avg_accuracy = total_accuracy / len(self.eval_dataloader)
        avg_reward_diff = total_reward_diff / len(self.eval_dataloader)
        
        # 恢复为训练模式
        self.model.train()
        
        logger.info(
            f"评估 - Loss: {avg_loss:.6f}, Accuracy: {avg_accuracy:.4f}, "
            f"Reward Diff: {avg_reward_diff:.4f}"
        )
        
        return {
            "eval_loss": avg_loss,
            "eval_accuracy": avg_accuracy,
            "eval_reward_diff": avg_reward_diff
        }
    
    def save_model(self, path: str) -> None:
        """
        保存模型
        
        参数:
            path: 保存路径
        """
        # 创建目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存检查点
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "args": self.args,
            "vocab_size": getattr(self.model, "vocab_size", None),
            "hidden_size": getattr(self.model, "hidden_size", None),
            "train_losses": self.train_losses,
            "eval_losses": self.eval_losses,
            "learning_rates": self.learning_rates,
            "preference_accuracies": self.preference_accuracies
        }, path)
        
        logger.info(f"模型已保存到 {path}")
    
    def load_model(self, path: str) -> None:
        """
        加载模型
        
        参数:
            path: 加载路径
        """
        # 检查文件存在
        if not os.path.exists(path):
            logger.error(f"模型文件不存在: {path}")
            return
        
        # 加载检查点
        checkpoint = torch.load(path, map_location=self.device)
        
        # 加载模型状态
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # 加载优化器状态
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # 加载训练历史
        if "train_losses" in checkpoint:
            self.train_losses = checkpoint["train_losses"]
        if "eval_losses" in checkpoint:
            self.eval_losses = checkpoint["eval_losses"]
        if "learning_rates" in checkpoint:
            self.learning_rates = checkpoint["learning_rates"]
        if "preference_accuracies" in checkpoint:
            self.preference_accuracies = checkpoint["preference_accuracies"]
        
        logger.info(f"模型已从 {path} 加载")
    
    def plot_training_curves(self) -> None:
        """绘制训练曲线"""
        plt.figure(figsize=(15, 10))
        
        # 绘制损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(self.train_losses, label="训练损失")
        plt.plot(self.eval_losses, label="评估损失")
        plt.xlabel("轮次")
        plt.ylabel("损失")
        plt.title("DPO 训练和评估损失")
        plt.legend()
        plt.grid(True)
        
        # 绘制学习率曲线
        plt.subplot(2, 2, 2)
        plt.plot(self.learning_rates)
        plt.xlabel("轮次")
        plt.ylabel("学习率")
        plt.title("学习率变化")
        plt.grid(True)
        
        # 绘制偏好准确率曲线
        plt.subplot(2, 2, 3)
        plt.plot(self.preference_accuracies)
        plt.xlabel("轮次")
        plt.ylabel("偏好准确率")
        plt.title("偏好准确率")
        plt.grid(True)
        
        # 保存图像
        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plt.savefig(os.path.join("plots", "dpo_training_curves.png"))
        plt.close() 