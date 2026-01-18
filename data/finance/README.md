# SpaceExploreAI 数据处理与模型训练指南

本指南详细介绍了 SpaceExploreAI 项目的完整工作流，包括数据下载、数据清洗、模型训练以及推理预测。

## 环境准备

在开始之前，请确保您位于项目根目录，并设置了 `PYTHONPATH`：

```bash
# 确保在项目根目录下
cd /Users/bytedance/AI/SpaceExploreAI

# 设置PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.
```

## 1. 数据下载 (Data Download)

使用 `download_data.py` 从 Yahoo Finance 下载股票历史价格数据。

```bash
# 下载指定股票数据 (推荐) one step
python data/finance/download_data.py --tickers AAPL,MSFT,GOOGL --start_date 2025-01-01

# 下载默认股票列表 (BILI, PDD, BABA, AMD, ASML, TSM, KO, F)
python data/finance/download_data.py

# 查看更多参数
python data/finance/download_data.py --help
```

**常用参数：**
- `--tickers`: 股票代码，多个用逗号分隔
- `--start_date`: 开始日期 (YYYY-MM-DD)
- `--output_dir`: 输出目录 (默认: `data/raw/price_history`)

## 2. 数据处理 (Data Processing)

使用 `process_downloaded_data.py` 对下载的数据进行清洗、特征工程和序列化处理，生成用于训练的 `.npz` 文件。

```bash
# 处理指定股票数据 (推荐) two steps
python data/finance/process_downloaded_data.py --tickers AAPL,MSFT,GOOGL

# 处理所有已下载的股票数据
python data/finance/process_downloaded_data.py

# 自定义序列长度和预测周期
python data/finance/process_downloaded_data.py --tickers AAPL --seq_length 60 --pred_horizon 5
```

**常用参数：**
- `--tickers`: 指定要处理的股票代码
- `--seq_length`: 输入序列长度 (默认: 32)
- `--pred_horizon`: 预测未来多少天 (默认: 2)
- `--test_size`: 测试集比例 (默认: 0.1)
- `--val_size`: 验证集比例 (默认: 0.1)

数据处理完成后，结果将保存在 `data/processed` 目录下：
- `train/`: 训练集数据
- `eval/`: 验证集数据
- `test/`: 测试集数据
- `scalers/`: 数据缩放器 (用于推理时的反归一化)

## 3. 模型训练 (Model Training)

使用 `train/train_model.py` 训练时间序列预测模型。

```bash
# 训练模型 (使用默认参数) three steps
python train/train_model.py --tickers AAPL,MSFT,GOOGL

# 自定义训练参数
python train/train_model.py --tickers AAPL --epochs 50 --batch_size 32 --learning_rate 0.0001 --sequence_length 60

# 合并多只股票数据进行联合训练
python train/train_model.py --tickers AAPL,MSFT,GOOGL --merge_stocks --epochs 100
```

**常用参数：**
- `--tickers`: 参与训练的股票代码
- `--epochs`: 训练轮数 (默认: 100)
- `--batch_size`: 批次大小 (默认: 64)
- `--learning_rate`: 学习率 (默认: 1e-4)
- `--sequence_length`: 序列长度 (需与数据处理阶段保持一致)
- `--merge_stocks`: 是否合并所有股票数据训练同一个模型

## 4. 推理预测 (Inference)

使用 `inference/run_inference.py` 加载训练好的模型进行预测。

```bash
# 单次预测
python inference/run_inference.py --tickers AAPL --model_path models/SpaceExploreAI_final.pt

# 批量预测多只股票
python inference/run_inference.py --tickers AAPL,MSFT --model_path models/SpaceExploreAI_final.pt

# 持续运行模式 (每小时运行一次)
python inference/run_inference.py --tickers AAPL --mode continuous --interval 3600
```

**常用参数：**
- `--tickers`: 需要预测的股票代码
- `--model_path`: 模型文件路径
- `--mode`: 运行模式 (`once` 或 `continuous`)
- `--output`: 结果输出文件路径 (可选)

## 目录结构说明

```
SpaceExploreAI/
├── data/
│   ├── finance/
│   │   ├── download_data.py         # 下载脚本
│   │   └── process_downloaded_data.py # 处理脚本
│   ├── raw/                         # 原始数据存储
│   └── processed/                   # 处理后数据存储
├── train/
│   └── train_model.py               # 训练脚本
├── inference/
│   └── run_inference.py             # 推理脚本
└── models/                          # 训练好的模型保存目录
```
