# SpaceExploreAI 实时股票预测使用指南

本文档介绍如何使用 SpaceExploreAI 模型进行实时股票预测，通过加载预训练模型，获取当天最新股票数据，预测明天的股价收益率。

## 准备工作

在使用预测脚本前，请确保您已经：

1. 安装了所有必要的依赖库：
   ```bash
   pip install torch numpy pandas matplotlib yfinance scikit-learn tqdm
   ```

2. 确保模型文件路径正确 (`models/SpaceExploreAI_best.pt` 或您指定的路径)

## 使用方法

### 基本用法

预测指定股票明天的收益率：

```bash
python inference.py --ticker AAPL
```

这将使用默认的模型文件 (`models/SpaceExploreAI_best.pt`) 预测 Apple 股票明天的收益率。

### 高级选项

您可以通过以下选项自定义预测过程：

```bash
python inference.py --ticker AAPL --model models/SpaceExploreAI_best.pt --seq_len 60 --horizon 1 --visualize
```

参数说明：
- `--ticker`: 股票代码 (例如：AAPL, MSFT, GOOGL)
- `--model`: 模型文件路径
- `--seq_len`: 输入序列长度（使用多少天的历史数据作为输入）
- `--horizon`: 预测周期（预测未来多少天）
- `--visualize`: 是否可视化预测结果（添加此选项将显示预测图表）

### 预测结果说明

脚本将输出以下信息：

#### 回归模型（预测确切收益率）
```
==================================================
  AAPL 股票预测结果
==================================================
预测日期: 2024-05-09
目标日期: 2024-05-10
最新价格: 185.59
预测收益率: 0.35%
预测价格: 186.24
==================================================
```

#### 分类模型（预测上涨/下跌概率）
```
==================================================
  AAPL 股票预测结果
==================================================
预测日期: 2024-05-09
目标日期: 2024-05-10
上涨概率: 0.65
下跌概率: 0.30
横盘概率: 0.05
预测结果: 上涨
==================================================
```

### 可视化结果

如果使用 `--visualize` 选项，脚本将生成预测可视化图表，并保存到 `results/predict/` 目录下。

图表包含：
- 最近30天的历史价格走势
- 预测的明天价格或价格变动方向
- 预测的收益率或上涨/下跌概率

## 在其他脚本中使用预测功能

您也可以在自己的Python脚本中导入并使用预测器：

```python
from inference import StockPredictor

# 初始化预测器
predictor = StockPredictor(
    model_path="models/SpaceExploreAI_best.pt",
    sequence_length=60,
    prediction_horizon=1
)

# 预测单只股票
result = predictor.predict("AAPL")
print(f"预测收益率: {result['predicted_return_percent']:.2f}%")

# 可视化预测结果
predictor.visualize_prediction("AAPL")
```

## 常见问题

### 1. 找不到模型文件
确保模型文件路径正确，默认路径是 `models/SpaceExploreAI_best.pt`。您也可以通过 `--model` 参数指定其他路径。

### 2. 获取股票数据失败
可能是网络问题或者Yahoo Finance API的限制。请检查：
- 网络连接是否正常
- 股票代码是否正确
- 是否超过了API请求限制

### 3. 预测质量问题
预测质量取决于多种因素：
- 模型训练数据的质量和时效性
- 输入序列长度是否合适
- 特定股票的波动特性
- 市场整体环境

请注意，金融市场预测存在内在不确定性，请将预测结果作为参考，而非投资决策的唯一依据。

## 后续优化方向

1. 添加更多数据源支持（不仅限于Yahoo Finance）
2. 支持批量预测多支股票
3. 集成外部市场情绪指标
4. 添加模型解释功能，解释预测背后的主要因素
5. 提供Web界面进行交互式预测

## 免责声明

SpaceExploreAI股票预测工具仅供研究和参考使用，不构成任何投资建议。使用本工具进行的投资决策风险由用户自行承担。市场有风险，投资需谨慎。 