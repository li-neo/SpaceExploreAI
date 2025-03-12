# 股价预测模型训练指南

本文档提供了如何准备数据、训练和评估股票预测模型的详细步骤。

## 依赖

  requirements：

```bash
pip install pandas numpy torch matplotlib tqdm scikit-learn pandas_ta yfinance
```

## 数据准备

1. **获取股票数据**

   首先，需要获取股票历史价格数据。可以使用以下命令从Yahoo Finance获取数据：

   ```bash
   python -m SpaceExploreAI.data.download_data --tickers AAPL,MSFT,GOOGL,TSLA --start_date 2022-01-01 --end_date 2025-03-012 --source yahoo
   ```

   这将下载指定股票的历史价格数据，并保存到`data/raw`目录。

2. **处理数据**

   可以在训练脚本中自动处理数据，也可以使用以下命令提前处理：

   ```bash
   python -m SpaceExploreAI.data.process_data --tickers AAPL --feature_groups technical,time,lag,return --sequence_length 60
   ```

   这将处理数据并计算技术指标，然后将处理后的数据保存到`data/processed`目录。

## 模型训练

### 基本训练

使用以下命令启动基本训练：

```bash
python -m SpaceExploreAI.train.train_model --tickers AAPL --sequence_length 60 --prediction_horizon 5 --batch_size 32 --num_epochs 30
```

### 高级训练选项

以下是一些高级训练选项的示例：

1. **多股票联合训练**

   ```bash
   python -m SpaceExploreAI.train.train_model --tickers AAPL,MSFT,GOOGL --merge_stocks --sequence_length 60 --batch_size 64
   ```

2. **调整模型架构**

   ```bash
   python -m SpaceExploreAI.train.train_model --tickers AAPL --hidden_size 512 --num_layers 6 --num_heads 8
   ```

3. **调整MoE参数**

   ```bash
   python -m SpaceExploreAI.train.train_model --tickers AAPL --moe_intermediate_size 512 --num_experts 16 --num_experts_per_token 4
   ```

4. **分类任务（预测上涨/下跌）**

   ```bash
   python -m SpaceExploreAI.train.train_model --tickers AAPL --prediction_type classification
   ```

5. **从检查点恢复训练**

   ```bash
   python -m SpaceExploreAI.train.train_model --tickers AAPL --resume_from ./models/stock_transformer_best.pt
   ```

6. **使用已处理的数据**

   ```bash
   python -m SpaceExploreAI.train.train_model --tickers AAPL --load_processed
   ```

## 模型评估

模型训练完成后，可以通过以下方式在测试集上评估模型：

```bash
python -m SpaceExploreAI.evaluate.evaluate_model --model_path ./models/stock_transformer_best.pt --test_data AAPL
```

## 参数说明

训练脚本支持以下主要参数：

### 数据相关参数

- `--raw_data_dir`: 原始数据目录，默认`../data/raw`
- `--processed_data_dir`: 处理后数据目录，默认`../data/processed`  
- `--tickers`: 股票代码，多个用逗号分隔，默认`AAPL`
- `--data_source`: 数据源，如`yahoo`或`alphavantage`，默认`yahoo`
- `--load_processed`: 加载已处理的数据
- `--merge_stocks`: 合并多只股票的数据
- `--scaler_type`: 缩放器类型，如`standard`、`minmax`或`robust`，默认`robust`
- `--test_size`: 测试集比例，默认`0.1`
- `--val_size`: 验证集比例，默认`0.1`
- `--sequence_length`: 序列长度，默认`60`
- `--prediction_horizon`: 预测周期，默认`5`
- `--feature_groups`: 特征组，多个用逗号分隔
- `--batch_size`: 批量大小，默认`32`
- `--num_workers`: 数据加载线程数，默认`4`

### 模型相关参数

- `--hidden_size`: 隐藏层维度，默认`256`
- `--num_layers`: Transformer层数，默认`4`
- `--num_heads`: 注意力头数量，默认`4`
- `--qk_nope_head_dim`: 不使用旋转位置编码的Q/K头维度，默认`32`
- `--qk_rope_head_dim`: 使用旋转位置编码的Q/K头维度，默认`32`
- `--v_head_dim`: 值向量的头维度，默认`64`
- `--moe_intermediate_size`: MoE中间层维度，默认`256`
- `--num_experts`: 专家数量，默认`8`
- `--num_experts_per_token`: 每个token使用的专家数量，默认`2`
- `--attention_dropout`: 注意力Dropout比率，默认`0.1`
- `--hidden_dropout`: 隐藏层Dropout比率，默认`0.1`
- `--disable_mixed_attention`: 禁用混合潜在注意力
- `--prediction_type`: 预测类型，`regression`或`classification`，默认`regression`

### 训练相关参数

- `--learning_rate`: 学习率，默认`1e-4`
- `--weight_decay`: 权重衰减，默认`0.01`
- `--clip_grad_norm`: 梯度裁剪范数，默认`1.0`
- `--num_epochs`: 训练轮次，默认`30`
- `--patience`: 早停耐心，默认`5`
- `--save_dir`: 模型保存目录，默认`./models`
- `--model_name`: 模型名称，默认`stock_transformer`
- `--log_interval`: 日志记录间隔，默认`10`
- `--device`: 训练设备，默认`cuda`
- `--disable_mixed_precision`: 禁用混合精度训练
- `--resume_from`: 从检查点恢复训练
- `--seed`: 随机种子，默认`42` 