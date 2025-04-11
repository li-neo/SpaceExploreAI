# SpaceExploreAI DPO强化训练模块

本模块实现了Direct Preference Optimization (DPO)算法，用于进一步优化SpaceExploreAI模型的预测性能，

## 简介

DPO是一种无需显式训练奖励模型的强化学习方法，它直接通过偏好数据（好/坏样本对）优化模型。与传统的RLHF（Reinforcement Learning from Human Feedback）相比，DPO具有更简单的训练流程和更稳定的训练过程。

论文参考：[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023)

## 目录结构

```
RLHF/
├── README.md           # 本文档
├── dpo_args.py         # DPO参数配置类
├── data_processor.py   # 偏好数据处理模块
├── dpo_trainer.py      # DPO训练器实现
└── train_dpo.py        # DPO训练启动脚本
```

## 安装依赖

本模块依赖于主项目的环境。确保已安装所有主项目依赖。

## 使用方法

### 1. 准备偏好数据

DPO训练需要偏好数据，包含三元组(输入序列, 优质输出, 劣质输出)。可以通过以下两种方式获取：

**A. 使用合成数据**：
```bash
python -m RLHF.train_dpo --reference_model_path models/SpaceExploreAI_best.pt --create_synthetic_data --num_synthetic_samples 1000
```

**B. 使用自定义偏好数据**：

准备偏好数据集，保存为JSONL格式，每行包含：
```json
{"prompt": [...], "chosen": [...], "rejected": [...]}
```

### 2. 启动DPO训练

```bash
python -m RLHF.train_dpo \
    --reference_model_path models/SpaceExploreAI_best.pt \
    --preference_data_path data/preferences \
    --beta 0.1 \
    --dpo_batch_size 4 \
    --dpo_learning_rate 5e-6 \
    --dpo_num_epochs 3 \
    --save_dir models/dpo \
    --model_name SpaceExploreAI_DPO
```

### 3. 参数说明

主要参数解释：

- `--reference_model_path`: 参考模型路径（通常是SFT模型）
- `--preference_data_path`: 偏好数据路径
- `--beta`: KL散度正则化系数，控制偏离参考模型的程度
- `--dpo_batch_size`: 训练批次大小
- `--dpo_learning_rate`: 学习率
- `--dpo_num_epochs`: 训练轮数
- `--create_synthetic_data`: 是否创建合成偏好数据
- `--num_synthetic_samples`: 合成样本数量

完整参数列表请参考`dpo_args.py`。

## DPO原理简述

DPO算法通过以下步骤工作：

1. 使用参考模型（SFT模型）作为起点
2. 对于每个偏好对(胜者,败者)，计算当前策略和参考策略的对数概率差
3. 应用专门设计的损失函数，使模型在保持接近参考模型的同时，增加胜者概率、降低败者概率
4. DPO损失函数数学表达式：
   ```
   L_DPO = -E_{(x,y_w,y_l)~D} [log(σ(β * (r_θ(x,y_w) - r_θ(x,y_l))))]
   ```
   其中r_θ是隐式奖励函数，由策略模型p_θ和参考模型p_ref的对数概率差定义：
   ```
   r_θ(x,y) = β * log(p_θ(y|x) / p_ref(y|x))
   ```

## 性能评估

DPO训练过程会跟踪以下指标：

- **损失**：DPO损失值
- **偏好准确率**：模型正确识别优质输出的比例
- **奖励差**：优质输出和劣质输出之间的奖励差值

训练完成后，会在`plots/`目录生成训练曲线图表，帮助评估训练效果。

## 模型保存与加载

训练期间会自动保存以下检查点：
- 每轮epoch结束：`{save_dir}/{model_name}_epoch{i}.pt`
- 验证集最佳性能：`{save_dir}/{model_name}_best.pt`
- 训练结束：`{save_dir}/{model_name}_final.pt`

## 常见问题

**Q: 什么是合适的beta值？**
A: beta控制KL散度正则化强度，较大的值(>0.5)会让模型保持接近参考模型，较小的值(<0.1)允许更大偏差。典型值为0.1-0.2。

**Q: 如何判断DPO训练是否有效？**
A: 观察偏好准确率是否提高（>70%表示良好），同时验证损失下降。

**Q: 为什么需要参考模型？**
A: 参考模型作为锚点，防止策略过度偏离原始分布，导致过拟合或功能退化。 