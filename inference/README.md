# 股票价格预测实时推理系统

本目录包含用于实时预测股票价格的推理代码，可以加载训练好的模型，下载最新的股票数据，并进行实时推理。

## 文件说明

- `inferencer.py`: 核心推理类，负责加载模型、下载和处理数据、执行预测
- `run_inference.py`: 命令行接口，用于运行推理，支持单次和持续运行模式

## 使用方法

### 基本用法

```bash
# 使用默认参数进行单次预测
python inference/run_inference.py

# 指定多只股票进行预测
python inference/run_inference.py --tickers "AAPL,MSFT,GOOG"

# 指定模型路径
python inference/run_inference.py --model_path "models/SpaceExploreAI_best.pt"
```

### 持续运行模式

```bash
# 每小时自动更新数据并预测
python inference/run_inference.py --mode continuous --interval 3600

# 每10分钟更新一次并保存结果
python inference/run_inference.py --mode continuous --interval 600 --output "results/predictions.json"
```

### 完整参数说明

```
usage: run_inference.py [-h] [--tickers TICKERS] [--model_path MODEL_PATH]
                        [--device DEVICE] [--raw_data_dir RAW_DATA_DIR]
                        [--processed_data_dir PROCESSED_DATA_DIR]
                        [--feature_groups FEATURE_GROUPS]
                        [--sequence_length SEQUENCE_LENGTH]
                        [--prediction_horizon PREDICTION_HORIZON]
                        [--scaler_type SCALER_TYPE]
                        [--mode {once,continuous}] [--interval INTERVAL]
                        [--output OUTPUT]

股票价格预测实时推理

可选参数:
  -h, --help            显示帮助信息并退出
  --tickers TICKERS     要预测的股票代码，多个用逗号分隔，如'AAPL,MSFT,GOOG'
  --model_path MODEL_PATH
                        模型权重路径
  --device DEVICE       运行设备，如'cpu', 'cuda', 'mps'，默认自动选择
  --raw_data_dir RAW_DATA_DIR
                        原始数据目录
  --processed_data_dir PROCESSED_DATA_DIR
                        处理后数据目录
  --feature_groups FEATURE_GROUPS
                        特征组，多个用逗号分隔
  --sequence_length SEQUENCE_LENGTH
                        序列长度
  --prediction_horizon PREDICTION_HORIZON
                        预测周期
  --scaler_type SCALER_TYPE
                        缩放器类型
  --mode {once,continuous}
                        运行模式，'once'表示单次运行，'continuous'表示持续运行
  --interval INTERVAL   连续模式下的更新间隔（秒）
  --output OUTPUT       输出结果到文件
```

## 关键说明

1. **模型与训练保持一致性**：
   - 推理系统使用了与训练相同的特征集 (time, lag, return, volatility, volume)
   - 保持了相同的序列长度 (32) 和预测周期 (2)
   - 使用了相同的数据处理流程和特征维度 (64)

2. **数据处理流程**：
   - 使用 `download_data.py` 下载最新的股票数据
   - 使用 `process_downloaded_data.py` 处理数据并添加特征
   - 确保处理后的特征维度与训练保持一致 (64)

3. **模型加载**：
   - 默认使用 `models/SpaceExploreAI_best.pt` 模型权重
   - 可以通过 `--model_path` 参数指定其他模型

## 示例输出

成功预测后，将输出类似以下格式的结果：

```json
[
  {
    "ticker": "AAPL",
    "prediction": 0.023,
    "timestamp": "2023-07-01 14:30:45"
  }
]
```

其中 `prediction` 是预测的未来收益率值。

## 故障排除

1. 如果遇到模型加载错误，请确保指定的模型路径正确，且使用了与训练时相同的模型结构。

2. 如果遇到数据下载或处理错误：
   - 检查网络连接
   - 确保 Yahoo Finance API 可用
   - 检查原始数据和处理后数据目录是否存在且有写权限

3. 如果预测结果不合理：
   - 检查特征维度是否与训练时一致
   - 确认使用了正确的模型权重
   - 验证数据处理过程中的特征组是否与训练时相同 