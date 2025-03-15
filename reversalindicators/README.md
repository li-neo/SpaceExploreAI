# 短线趋势反转交易策略

这个项目实现了一个基于技术指标、成交量分析、相对强弱和支撑/阻力位的短线趋势反转交易策略。该策略专为期权交易设计，可以帮助交易者识别市场顶部和底部的反转信号。

## 策略概述

该策略基于以下关键要素：

1. **背离分析**：识别价格与技术指标（RSI、MACD、随机指标）之间的背离
2. **成交量确认**：分析反转点附近的成交量放大情况
3. **相对强弱比较**：与大盘指数（如QQQ）的相对表现
4. **支撑/阻力位分析**：识别关键价格水平和最小阻力路径

策略通过综合评分系统（0-14分）来量化信号强度，并根据信号强度提供仓位建议和成功概率估计。

## 文件结构

- `trend_reversal_strategy.md`: 详细的策略文档，包括信号识别、评分系统和交易执行指南
- `trend_reversal_calculator.py`: Python实现的信号计算器，可以自动分析股票数据并生成交易信号
- `indicators.md`: 原始策略要点概述

## 安装与使用

### 安装依赖

```bash
pip install -r requirements.txt
```

注意：TA-Lib可能需要额外的安装步骤，请参考[TA-Lib安装指南](https://github.com/mrjbq7/ta-lib#installation)。

### 使用示例

```python
import pandas as pd
import yfinance as yf
from trend_reversal_calculator import TrendReversalSignal

# 下载股票数据
ticker = "NVDA"
ref_ticker = "QQQ"

# 获取分钟级数据（最近5天）
stock_data = yf.download(ticker, period="5d", interval="1m")
ref_data = yf.download(ref_ticker, period="5d", interval="1m")

# 初始化信号计算器
signal = TrendReversalSignal(ticker, ref_ticker)
signal.load_data(stock_data, ref_data)

# 计算信号强度
result = signal.calculate_signal_strength()

# 打印结果
print(f"信号类型: {result['signal_type']}")  # CALL 或 PUT
print(f"信号强度: {result['signal_strength']}/14")
print(f"信号等级: {result['signal_class']}")
print(f"成功概率: {result['success_probability']}")
print(f"建议仓位: {result['recommended_position']}")

# 可视化信号
signal.plot_signal()
```

## 信号强度评分系统

信号强度基于四个关键因素的评分：

1. **背离评分** (0-4分)
   - 单一指标背离: +1分
   - 两个指标同时背离: +2分
   - 三个或更多指标同时背离: +3分
   - 背离幅度超过20%: 额外+1分

2. **成交量评分** (0-4分)
   - 成交量放大3-4倍: +1分
   - 成交量放大4-5倍: +2分
   - 成交量放大5倍以上: +3分
   - 成交量持续时间超过3根K线: 额外+1分

3. **相对强弱评分** (0-3分)
   - 轻微强/弱 (相对强度偏离5-10%): +1分
   - 中等强/弱 (相对强度偏离10-15%): +2分
   - 显著强/弱 (相对强度偏离>15%): +3分

4. **支撑/阻力评分** (0-3分)
   - 单一支撑/阻力位: +1分
   - 多重支撑/阻力位重叠: +2分
   - 历史关键价位+高成交量区域重叠: +3分

### 信号强度分级

- **弱信号** (1-4分): 观望或小仓位试探，成功概率<40%
- **中等信号** (5-8分): 适中仓位，严格止损，成功概率40-60%
- **强信号** (9-11分): 较大仓位，适当追踪止损，成功概率60-75%
- **极强信号** (12-14分): 满仓操作，宽松追踪止损，成功概率>75%

## 风险管理

- 单笔交易风险不超过总资金的2%
- 根据信号强度调整仓位
- 连续亏损3次后，强制休息1-2天重新评估
- 大盘极端波动日（VIX>30）减半仓位或暂停交易

## 注意事项

- 该策略主要适用于短线期权交易
- 对于大科技股，需要考虑与QQQ的相对强弱
- 止损点设置在0.5个点（期权价格变动）
- 始终沿着市场最小阻力位进行操作 