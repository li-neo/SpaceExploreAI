# SpaceExploreAI 投资预测大模型

## 项目介绍

SpaceExploreAI 是一个基于深度学习的投资预测系统，采用最先进的 Transformer 架构设计，更是吸取了DeepSeep-V3、LLama3的多种设计结构，如MLA、MOE，可用于股票价格趋势预测、技术分析和投资决策辅助。该项目结合了现代深度学习技术和传统金融分析方法，包括技术指标、时间序列分析等，提供了一个全面的股票预测解决方案。

## 免责声明
SpaceExploreAI仅供大模型AI学习、量化交易学习，不可以用于商业用途、不可以以此为投资逻辑，后果自负。

## License
[MIT License](LICENSE)

### 主要特点

- **先进的模型架构**：基于 Transformer 架构，集成了 MLA 多头潜在注意力、RoPE 旋转位置编码、MoE 混合专家和残差连接等先进技术
- **丰富的技术指标**：内置超过 30 种技术分析指标，包括移动平均线、RSI、MACD、布林带等
- **灵活的训练配置**：支持多种训练参数配置，包括回归和分类任务、多股票联合训练等
- **完整的数据流水线**：自动化的数据获取、预处理、特征工程和模型训练流程
- **详细的模型评估**：提供多种评估指标和可视化方法，帮助理解模型性能

## 系统架构

```
SpaceExploreAI/
├── data/                  # 数据相关模块
│   ├── download_data.py   # 股票数据下载工具
│   ├── data_processor.py  # 数据处理工具
│   └── technical_indicators.py  # 技术指标计算
├── model/                 # 模型定义
│   ├── transformer.py     # Transformer 模型架构
│   ├── attention.py       # 注意力机制实现
│   └── moe.py             # 混合专家模型实现
├── train/                 # 训练相关
│   ├── train_model.py     # 训练脚本
│   ├── trainer.py         # 训练器实现
│   └── README.md          # 训练指南
├── evaluate/              # 评估工具
│   └── evaluate_model.py  # 模型评估脚本
└── examples/              # 示例
    └── quick_start.py     # 快速入门示例
```

## 模型架构

该项目的核心是一个基于 Transformer 的股价预测模型，其主要组件包括：

1. **Tokenizer**：进行数据采样、加工和编码
2. **Transformer**：借鉴 DeepSeek 和 LLama3 大模型的架构设计
3. **MLA**：多头潜在注意力机制，增强模型对时间序列数据的理解
4. **RoPE**：旋转位置编码，帮助模型更好地理解序列位置信息
5. **MOE**：混合专家模块，通过多个专家网络提高模型表达能力
6. **MLP**：多层感知器，用于特征转换
7. **残差连接和层归一化**：确保深层网络的有效训练

### 技术细节

#### 多头潜在注意力 (MLA)

SpaceExploreAI 使用了创新的多头潜在注意力机制，将传统的自注意力扩展为同时支持旋转位置编码 (RoPE) 和无位置编码的注意力头，能够更好地捕捉时间序列数据中的长期和短期依赖关系。 后续版本在旋转位置编码的基础上，加权一个时间序列位置编码，原因是开盘前后的时间对特征向量具有较大影响值

```
Q, K, V 分解为：
- Q_rope, K_rope: 使用旋转位置编码的查询和键向量
- Q_nope, K_nope: 不使用位置编码的查询和键向量
- V: 值向量
```

#### 混合专家模型 (MoE)

系统使用的 MoE 架构包含多个并行的前馈网络（"专家"），由一个路由网络动态地为每个输入选择最合适的专家组合：

- **专家数量**: 默认 8 个专家
- **每个 token 使用的专家数**: 默认 2 个
- **路由算法**: 基于 Top-K 选择的稀疏 MoE

#### 优化技术

- **混合精度训练**：利用 FP16/BF16 加速训练过程（等后续添加DeepSeek FP8）
- **梯度累积**：支持更大的有效批量大小
- **早停机制**：防止过拟合
- **学习率调度**：使用 ReduceLROnPlateau 策略

## 安装指南

### 依赖项

```bash
pip install pandas numpy torch matplotlib tqdm scikit-learn pandas_ta yfinance seaborn
```

### 从源码安装

```bash
git clone https://github.com/yourusername/SpaceExploreAI.git
cd SpaceExploreAI
pip install -e .
```

## 快速入门

使用提供的快速入门脚本可以轻松完成从数据下载到模型评估的完整流程：

```bash
python -m SpaceExploreAI.examples.quick_start --tickers AAPL --num_epochs 10
```

这个命令将：
1. 下载 Apple 股票的历史数据
2. 处理数据并计算技术指标
3. 训练预测模型
4. 评估模型性能并生成可视化结果

## 详细使用指南

### 1. 数据获取

```bash
python -m SpaceExploreAI.data.download_data --tickers AAPL,MSFT,GOOGL --start_date 2010-01-01
```

#### 支持的数据源

目前系统主要支持从 Yahoo Finance 获取数据，后续将支持更多数据源：

- **Yahoo Finance**: 免费的历史股票数据
- **未来计划支持**: AlphaVantage, Quandl, IEX Cloud 等

#### 数据目录结构

下载的原始数据将保存在以下结构：

```
data/raw/price_history/
├── AAPL/
│   └── AAPL_yahoo_20230301.csv
├── MSFT/
│   └── MSFT_yahoo_20230301.csv
└── ...
```

### 2. 数据处理流程

股票数据在训练前需要经过一系列处理步骤：

1. **加载原始价格数据**：从 CSV 文件加载 OHLCV (开盘价、最高价、最低价、收盘价、成交量) 数据
2. **清洗数据**：处理缺失值、异常值和时间序列缺口
3. **计算技术指标**：使用 `technical_indicators.py` 计算超过 30 种技术指标
4. **特征选择**：可以通过 `--feature_groups` 参数选择特定特征组合
5. **特征缩放**：使用 Robust/Standard/MinMax 缩放器标准化特征
6. **序列创建**：根据 `sequence_length` 和 `prediction_horizon` 构建输入-输出序列对
7. **训练-验证-测试拆分**：默认比例为 80%-10%-10%

### 3. 模型训练

基本训练：

```bash
python -m SpaceExploreAI.train.train_model --tickers AAPL --sequence_length 60 --prediction_horizon 5 --num_epochs 30
```

多股票联合训练：

```bash
python -m SpaceExploreAI.train.train_model --tickers AAPL,MSFT,GOOGL --merge_stocks
```

分类任务（预测上涨/下跌）：

```bash
python -m SpaceExploreAI.train.train_model --tickers AAPL --prediction_type classification
```

### 4. 模型评估

```bash
python -m SpaceExploreAI.evaluate.evaluate_model --model_path ./models/stock_transformer_best.pt --test_data AAPL
```

## 技术指标

系统内置的技术指标包括：

- 移动平均线 (MA, EMA)
- 相对强弱指数 (RSI)
- 随机振荡器 (Stochastic Oscillator)
- MACD (Moving Average Convergence Divergence)
- 布林带 (Bollinger Bands)
- 平均真实范围 (ATR)
- 能量潮 (OBV)
- 一目均衡表 (Ichimoku Cloud)
- 斐波那契回调水平
- 价格通道
- 波动率指标
- 动量指标

## 高级用法

### 调整模型架构

```bash
python -m SpaceExploreAI.train.train_model --hidden_size 512 --num_layers 6 --num_heads 8
```

### 调整混合专家参数

```bash
python -m SpaceExploreAI.train.train_model --moe_intermediate_size 512 --num_experts 16 --num_experts_per_token 4
```

### 从检查点恢复训练

```bash
python -m SpaceExploreAI.train.train_model --resume_from ./models/stock_transformer_last.pt
```

## 模型效果

该模型针对不同股票和市场条件进行了广泛测试，展现了良好的预测能力：

- **回归任务**：预测未来价格变动的方向和幅度
- **分类任务**：预测股价上涨/下跌/横盘的概率

典型的评估指标包括：
- MSE (均方误差)
- RMSE (均方根误差)
- MAE (平均绝对误差)
- 方向准确率 (股价涨跌方向预测的准确性)

## 项目愿景

根据项目介绍文档，SpaceExploreAI 股价预测大模型的最终目标是：

1. 实现类似大型语言模型的直接交互能力，用户可以直接询问股票相关问题
2. 具备实时获取最新股票信息并进行技术分析、基本面分析和量化分析的能力
3. 提供实时的投资建议和投资策略

当前版本专注于时间序列预测能力，未来版本将进一步增强交互性和实时分析能力。

## 贡献指南

欢迎对项目进行贡献！可以通过以下方式参与：

1. 提交 bug 报告或功能请求
2. 提交代码改进或新功能
3. 改进文档


## 案例分析

### 案例 1: 苹果股票价格预测

以下是使用模型预测 Apple 公司股票未来 5 天价格变动的案例：

#### 训练命令

```bash
python -m SpaceExploreAI.train.train_model --tickers AAPL --sequence_length 60 --prediction_horizon 5 --hidden_size 256 --num_layers 4 --num_heads 4 --num_epochs 30
```

#### 关键结果

- **MSE**: 0.0023
- **方向准确率**: 68.7%
- **平均训练时间**: 15 分钟 (单 GPU)

### 案例 2: 多股票联合训练

通过联合训练 FAANG 股票（Facebook/Meta, Apple, Amazon, Netflix, Google）提高模型泛化能力：

```bash
python -m SpaceExploreAI.train.train_model --tickers META,AAPL,AMZN,NFLX,GOOGL --merge_stocks --num_epochs 50
```

联合训练的模型在单股票上的表现比单独训练的模型有显著提升，特别是在市场波动较大的时期。

## 自定义模型和扩展

### 添加新的技术指标

可以通过扩展 `TechnicalIndicatorProcessor` 类添加自定义技术指标：

```python
# SpaceExploreAI/data/technical_indicators.py
class CustomIndicatorProcessor(TechnicalIndicatorProcessor):
    def __init__(self):
        super().__init__()
        
    def add_custom_indicator(self, df):
        # 实现自定义指标计算
        df['custom_indicator'] = ...
        return df
        
    def calculate_all_indicators(self, df):
        df = super().calculate_all_indicators(df)
        df = self.add_custom_indicator(df)
        return df
```

### 创建自定义预测任务

除了默认的价格预测，您还可以定义自己的预测任务，如预测波动率、交易量或特定技术指标：

```python
# 示例: 创建波动率预测任务
def create_volatility_targets(price_data, window=21):
    returns = price_data['close'].pct_change()
    volatility = returns.rolling(window=window).std() * np.sqrt(252)
    return volatility.shift(-1)  # 预测下一天的波动率
```

## 性能调优

### 硬件推荐

- **训练**: GPU 内存 ≥ 8GB (推荐 NVIDIA RTX 3060 或更高)
- **推理**: CPU 或 GPU 均可
- **内存**: ≥ 16GB RAM


### 超参数调优

以下是一些关键超参数及其调整建议：

| 参数 | 说明 | 推荐范围 | 影响 |
|------|------|----------|------|
| learning_rate | 学习率 | 1e-5 ~ 1e-3 | 收敛速度和稳定性 |
| hidden_size | 隐藏层维度 | 128 ~ 512 | 模型容量和训练速度 |
| num_layers | Transformer 层数 | 2 ~ 8 | 模型深度和复杂度 |
| num_heads | 注意力头数量 | 4 ~ 16 | 并行特征学习能力 |
| num_experts | MoE 专家数量 | 4 ~ 32 | 模型容量和计算复杂度 |
| sequence_length | 输入序列长度 | 30 ~ 120 | 历史依赖性和训练效率 |

## 实时预测与部署

### 部署模型 API

可以使用 Flask/FastAPI 创建模型 API 服务：

```python
from flask import Flask, request, jsonify
from SpaceExploreAI.train.trainer import create_stock_predictor_from_checkpoint

app = Flask(__name__)
predictor = create_stock_predictor_from_checkpoint("./models/stock_transformer_best.pt")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    ticker = data['ticker']
    # 获取最新数据并处理
    # ...
    prediction = predictor.predict(processed_data)
    return jsonify({"prediction": prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 实时数据集成

可以使用第三方 API 或数据服务来获取实时市场数据，与模型进行集成：

```python
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_latest_data(ticker, sequence_length=60):
    # 获取最新市场数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=sequence_length * 2)  # 获取足够的历史数据
    
    data = yf.download(ticker, start=start_date, end=end_date)
    # 处理数据并计算技术指标
    # ...
    return processed_data
```

## 常见问题解答

### Q: 如何处理缺失的股票价格数据？

A: 系统默认使用前向填充 (Forward Fill) 处理交易日内的缺失值。对于长期缺失数据，建议获取完整的数据源或使用插值方法。可以通过扩展 `StockDataProcessor` 类来自定义缺失值处理策略。

### Q: 回归预测和分类预测有什么区别？

A: 
- **回归预测**: 直接预测未来价格变动的幅度（连续值）
- **分类预测**: 预测价格变动的方向（上涨、下跌、横盘）

回归预测适合需要精确价格变动的场景，而分类预测适合只关注变动方向的交易策略。

### Q: 模型是否支持季节性和周期性特征？

A: 是的。多头潜在注意力和旋转位置编码使模型能够捕捉时间序列中的周期性模式。此外，技术指标处理器还添加了时间特征（如星期几、月份、季度等），帮助模型识别季节性因素。

### Q: 如何解读模型的预测结果？

A: 预测结果通常为归一化的价格变动或概率分布：

- **回归预测**: 返回归一化的未来收益率，需要使用相同的缩放器反向转换为实际价格变动
- **分类预测**: 返回各类别的概率分布，例如 [0.2, 0.7, 0.1] 表示下跌、上涨、横盘的概率

可以使用以下代码反向转换回实际价格变动：

```python
# 回归预测反向转换
actual_change = processor.scalers[ticker]['target'].inverse_transform(prediction)
next_price = current_price * (1 + actual_change)
```

## 研究与引用

如果您在研究中使用了本项目，请引用：

```
@misc{SpaceExploreAI2025,
  author = {li-neo&chang zunling},
  title = {SpaceExploreAI: A Transformer-based Stock Price Prediction System},
  year = {2025},
  publisher = {GitHub & Hugging Face},
  url = {https://github.com/li-neo/SpaceExploreAI}
}
```

## 版本历史

- **v1.0.0** (2025-03-12): 初始版本，包含核心预测功能
- **v0.3.0** (2024-12-07): Beta 版本，优化大模型注意力模型，改为MLA、增加多专家MOE、添加API、UI、测试样本
- **v0.2.0** (2024-06-12): Beta 版本，添加Transformer模块、数据导入、Embedding、清洗
- **v0.1.0** (2024-02-01): Beta 版本，版本规划、需求、设计

## 社区与支持

- **问题报告**: 请在 [GitHub Issues](https://github.com/li-neo/SpaceExploreAI/issues) 提交问题
- **联系邮箱**: liguangxian1995@gmail.com 

## 鸣谢

感谢项目合伙人常遵领的日夜奋战、感谢阿姣同学在投资策略上给予的支持和帮助，以及 DeepSeek、LLama3、ChatGPT、Hugging Face在AI领域的探索。