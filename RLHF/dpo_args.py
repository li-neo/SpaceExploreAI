from typing import Literal
from dataclasses import dataclass, field
from train.model_args import ModelArgs

@dataclass
class DPOArgs(ModelArgs):
    """DPO强化学习参数配置

    Direct Preference Optimization (DPO)是一种无需显式训练奖励模型的强化学习方法，
    通过直接从人类偏好数据中学习，优化模型生成令人满意的预测结果。
    
    DPO主要参数:
    1. beta: 调节KL散度正则化强度的参数，控制模型偏离参考策略(SFT模型)的程度
       - 较大的beta会使模型更接近参考策略
       - 较小的beta允许模型更大幅度偏离参考策略，更好地拟合人类偏好
    2. 偏好数据: 包含(输入, 胜者输出, 败者输出)三元组的数据
    """
    # DPO 基本参数
    beta: float = 0.1  # KL散度正则化系数，控制偏离SFT模型的程度
    reference_model_path: str = None  # 参考模型路径（通常是SFT模型）
    preference_data_path: str = "data/preferences"  # 偏好数据路径
    max_prompt_length: int = 512  # 最大输入序列长度
    max_response_length: int = 128  # 最大响应序列长度
    
    # DPO训练参数
    dpo_batch_size: int = 4  # DPO训练批次大小
    dpo_learning_rate: float = 5e-6  # DPO训练学习率
    dpo_num_epochs: int = 3  # DPO训练轮数
    dpo_gradient_accumulation_steps: int = 4  # 梯度累积步数
    dpo_max_grad_norm: float = 0.5  # 梯度裁剪范数
    dpo_warmup_ratio: float = 0.1  # 预热比例
    
    # LoRA参数 (低秩适应，用于高效微调大模型)
    use_lora: bool = True  # 是否使用LoRA
    lora_rank: int = 8  # LoRA秩
    lora_alpha: int = 16  # LoRA缩放系数
    lora_dropout: float = 0.05  # LoRA丢弃率
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])  # 目标模块
    
    # 混合精度训练
    dpo_mixed_precision: str = "bf16"  # 混合精度类型：无、fp16、bf16或fp8
    
    # 评估参数
    eval_ratio: float = 0.1  # 评估数据比例
    eval_every: int = 100  # 每多少步评估一次
    
    # 偏好学习特定参数
    loss_type: str = "sigmoid"  # 损失类型：sigmoid或hinge
    reference_free: bool = False  # 是否使用无参考DPO
    reward_model_path: str = None  # 可选的奖励模型路径（用于混合DPO+RM训练）
    rm_weight: float = 0.0  # 奖励模型权重（0表示纯DPO） 