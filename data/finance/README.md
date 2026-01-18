# 金融数据处理模块

本模块负责股票数据的获取、处理和特征工程，为模型训练提供高质量的数据支持。

## 功能概述

1. **数据获取** (`download_data.py`)：
   - 从 Yahoo Finance 等数据源下载股票历史价格数据
   - 支持多只股票批量下载
   - 自动处理无效数据行

2. **数据处理** (`process_downloaded_data.py`)：
   - 加载原始 CSV 数据
   - 计算技术指标（SMA, RSI, MACD 等）
   - 生成时间序列特征
   - 数据标准化（RobustScaler, MinMaxScaler 等）
   - 划分训练集、验证集和测试集
   - 生成模型所需的序列数据（Sequence Data）

## 快速开始

### 1. 下载数据

使用 `download_data.py` 下载股票数据。

```bash
# 确保在项目根目录下，并设置PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# 下载默认股票数据 (BILI,PDD,BABA,AMD,ASML,TSM,KO,F)
python data/finance/download_data.py

# 下载指定股票数据
python data/finance/download_data.py --tickers AAPL,MSFT,GOOGL --start_date 2015-01-01

# 查看帮助
python data/finance/download_data.py --help
```

**参数说明：**
- `--tickers`: 股票代码，多个用逗号分隔 (默认: BILI,PDD,BABA,AMD,ASML,TSM,KO,F)
- `--start_date`: 开始日期 (默认: 2010-03-11)
- `--end_date`: 结束日期 (默认: 当前日期)
- `--output_dir`: 输出目录 (默认: ../data/raw)
- `--source`: 数据源 (默认: yahoo)
- `--interval`: 数据间隔 (默认: 1d)

### 2. 处理数据

使用 `process_downloaded_data.py` 清洗和处理数据，生成用于训练的 `.npz` 文件。

```bash
# 确保在项目根目录下，并设置PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# 处理所有已下载的股票数据
python data/finance/process_downloaded_data.py

# 处理指定股票数据，并设置序列长度和批次大小
python data/finance/process_downloaded_data.py --tickers AAPL,MSFT --seq_length 60 --batch_size 64

# 查看帮助
python data/finance/process_downloaded_data.py --help
```

**参数说明：**
- `--raw_dir`: 原始数据目录
- `--processed_dir`: 处理后数据目录
- `--tickers`: 指定要处理的股票代码，逗号分隔 (如果不指定，则处理所有下载的股票)
- `--test_size`: 测试集比例 (默认: 0.1)
- `--val_size`: 验证集比例 (默认: 0.1)
- `--seq_length`: 序列长度 (默认: 60)
- `--pred_horizon`: 预测周期 (默认: 1)
- `--batch_size`: 批处理大小 (默认: 64)
- `--scaler`: 数据标准化方式 (默认: robust)

## 数据目录结构

```
data/
├── raw/                      # 原始数据目录 (由 download_data.py 生成)
│   ├── price_history/       
│   │   ├── AAPL/           
│   │   │   └── AAPL_yahoo_20231027.csv
│   │   └── ...
│
├── processed/                # 处理后的数据目录 (由 process_downloaded_data.py 生成)
│   ├── train/               # 训练集序列数据 (*_train_sequences.npz)
│   ├── val/                 # 验证集序列数据 (*_val_sequences.npz)
│   ├── test/                # 测试集序列数据 (*_test_sequences.npz)
│   └── scalers/             # 数据缩放器 (*_scaler.pkl)
```

## 技术指标说明

数据处理过程中会自动计算以下技术指标：
- **Trend**: SMA, EMA, MACD
- **Momentum**: RSI
- **Volatility**: BB (Bollinger Bands), ATR
- **Volume**: OBV

## 常见问题

1. **ModuleNotFoundError**: 
   - 请确保在运行脚本前设置了 `PYTHONPATH`，例如：`export PYTHONPATH=$PYTHONPATH:.`

2. **下载失败**:
   - 检查网络连接
   - 确认股票代码是否正确
   - 尝试更新 `yfinance` 库: `pip install --upgrade yfinance`
