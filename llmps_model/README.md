# LLM-PS: Large Language Model for Time Series Forecasting

本模块实现了LLM-PS（Large Language Model for Time Series Forecasting）模型，该模型将大型语言模型与时间序列预测结合，通过多尺度卷积网络和时间到文本的语义提取来实现高精度时间序列预测。

## 主要特点

- **多尺度卷积网络（MSCNN）**: 使用小波变换和多尺度卷积，捕获时间序列中的短期和长期模式
- **时间到文本转换（T2T）**: 提取时间序列的语义表示，便于与LLM集成
- **跨模态融合**: 将时间序列特征与语义特征融合，提供更丰富的表示
- **预测解释**: 利用LLM生成关于预测结果的自然语言解释

## 项目结构

```
llm_ps/
├── configs/          # 配置文件
├── data/             # 数据处理模块
├── models/           # 模型定义
│   ├── mscnn.py      # 多尺度卷积网络
│   ├── t2t.py        # 时间到文本转换器
│   ├── llm_ps.py     # 完整LLM-PS模型
│   └── integration.py # 与SpaceExploreAI集成模块
├── utils/            # 工具函数
├── main.py           # 主脚本
└── train_integrated.py # 集成训练脚本
```

## 安装依赖

```bash
pip install -r llmps_model/requirements.txt
```

## 使用方法

### 1. 独立使用LLM-PS模型

#### 模型训练

```bash
python llmps_model/main.py --mode train --data_dir path/to/data --output_dir path/to/output
```

#### 模型评估

```bash
python llmps_model/main.py --mode eval --data_dir path/to/data --checkpoint_path path/to/checkpoint.pth --output_dir path/to/output
```

#### 模型推理

```bash
python llmps_model/main.py --mode inference --checkpoint_path path/to/checkpoint.pth --input_path path/to/input.npy --visualize
```

### 2. 与SpaceExploreAI集成的LLM-PS

#### 集成训练

```bash
python llmps_model/train_integrated.py --mode train --data_dir path/to/data --output_dir path/to/output --llm_checkpoint_path path/to/llm_model.pth
```

#### 集成推理

```bash
python llmps_model/train_integrated.py --mode inference --checkpoint_path path/to/integrated_model.pth --input_path path/to/input.npy --visualize
```

#### 生成预测和解释
集成模型可以同时生成时间序列预测和自然语言解释，使用以API:
   forecast, explanation = model.generate_forecast_with_explanation(time_series_input)

## 模型配置

可以通过JSON配置文件自定义模型参数：

```bash
python llmps_model/train_integrated.py --mode train --config_path path/to/config.json
```

配置文件示例：
```json
{
  "temporal_encoder_config": {
    "in_channels": 1,
    "base_channels": 64,
    "ms_blocks": 3,
    "scales": [3, 5, 7, 9],
    "seq_len": 96,
    "output_dim": 512,
    "use_decoupling": true
  },
  "semantic_encoder_config": {
    "in_channels": 1,
    "patch_size": 24,
    "overlap": 8,
    "embed_dim": 96,
    "num_encoder_layers": 4,
    "num_decoder_layers": 1,
    "nhead": 4,
    "dim_feedforward": 384,
    "dropout": 0.1,
    "mask_ratio": 0.75,
    "vocab_size": 5000,
    "output_dim": 512,
    "max_prompt_len": 50
  },
  "training": {
    "batch_size": 64,
    "epochs": 100,
    "learning_rate": 1e-4,
    "feature_weight": 0.5,
    "prompt_weight": 0.1,
    "llm_weight": 0.1
  }
}
```

## 损失函数

LLM-PS模型使用组合损失函数进行训练：

L_OBJ = L_TIME + λL_FEAT + μL_PROMPT + γL_LLM

其中：
- L_TIME: 时间序列预测损失（MSE）
- L_FEAT: 特征一致性损失，基于时间特征和语义特征之间的余弦相似度
- L_PROMPT: 提示词生成损失
- L_LLM: 语言模型损失

各损失组件权重（λ, μ, γ）可在配置文件中调整。

## 小波变换

模型使用小波变换（Wavelet Transform）进行时间序列分解，将信号分为低频（长期趋势）和高频（短期波动）组件：

```python
# 小波分解
coeffs = pywt.wavedec(signal, wavelet='db4', level=1)

# 低频重构
low_freq = pywt.waverec([coeffs[0]] + [None] * len(coeffs[1:]), 'db4')

# 高频重构
high_coeffs = [np.zeros_like(coeffs[0])]
for i, detail in enumerate(coeffs[1:]):
    high_coeffs.append(detail)
high_freq = pywt.waverec(high_coeffs, 'db4')
```

## 引用

如果您在研究中使用了LLM-PS模型，请引用以下论文：

```
@article{llm_ps2023,
  title={LLM-PS: Empowering Large Language Models for Time Series Forecasting with Temporal Patterns and Semantics},
  author={...},
  journal={...},
  year={2023}
}
``` 


# 小波分解说明（旧版本，现在已更新为按照num_branches进行时间模式解耦）

这段代码实现了时间模式解耦，使用小波变换将时间序列分解为低频和高频分量。让我详细解释这个过程及为什么要在batch和channels层分别进行小波变换：

### 代码详解

```python
# 对每个样本和通道分别进行小波变换
for b in range(batch_size):
    for c in range(channels):
        # 执行小波分解
        coeffs = pywt.wavedec(x_np[b, c], self.wavelet, level=self.level)
        
        # 分离低频和高频分量
        approx_coeffs = coeffs[0]  # 近似系数（低频）
        detail_coeffs = coeffs[1:]  # 细节系数（高频）
        
        # 重构低频分量（长期模式）
        # 保留近似系数，将细节系数设为0
        low_coeffs = [approx_coeffs] + [None] * len(detail_coeffs)
        low_freq[b, c] = pywt.waverec(low_coeffs, self.wavelet)[:seq_len]
        
        # 重构高频分量（短期模式）
        # 将近似系数设为0，保留细节系数
        high_coeffs = [np.zeros_like(approx_coeffs)]
        for i, detail in enumerate(detail_coeffs):
            high_coeffs.append(detail)
        high_freq[b, c] = pywt.waverec(high_coeffs, self.wavelet)[:seq_len]
        
        # 确保分解是完整的: 原始 = 低频 + 高频 (验证)
        residual = x_np[b, c] - (low_freq[b, c] + high_freq[b, c])
        if np.abs(residual).max() > 1e-10:
            # 如果残差较大，微调高频分量以确保精确分解
            high_freq[b, c] += residual
```

### 主要步骤：

1. **双重循环遍历**：逐个处理每个样本的每个通道
2. **小波分解**：使用`pywt.wavedec`函数对一维时间序列进行小波分解
3. **系数分离**：将分解结果分为近似系数(低频)和细节系数(高频)
4. **重构低频分量**：仅保留近似系数，重构得到长期趋势信号
5. **重构高频分量**：仅保留细节系数，重构得到短期波动信号
6. **误差修正**：检查并修正分解误差，确保原始信号等于低频+高频信号

### 为什么要分别在batch和channels层进行小波变换？

1. **小波变换的一维特性**：
   - 小波变换`pywt.wavedec`只能处理一维数组
   - 输入张量是三维的`[batch_size, channels, seq_len]`
   - 需要逐个提取每个一维时间序列进行处理

2. **保持特征独立性**：
   - 每个通道代表不同的特征(如股票的开盘价、收盘价等)
   - 不同特征有各自独特的时间模式，需要单独解耦
   - 混合处理会导致特征间信息交叉污染

3. **样本间无关联性**：
   - 批次中的不同样本通常表示不同实体或不同时间段
   - 样本间没有直接的时间相关性，不适合联合处理
   - 分别处理保持了样本间的独立性

4. **通道间的独立解耦**：
   - 每个通道可能有不同的频率特性和趋势特征
   - 某些通道可能主要包含低频信息，而其他通道可能主要有高频波动
   - 单独处理可以为每个通道找到最优的分解

5. **实现简单性**：
   - 使用双重循环处理三维张量的每个时间序列更直观
   - 避免复杂的批量处理可能引入的错误
   - 使得验证和调试更容易进行

### 实际意义举例

假设有金融数据：
- 16个不同股票(batch_size=16)
- 每个股票有5个特征：开盘价、收盘价、最高价、最低价、交易量(channels=5)
- 每个序列包含96个交易日数据(seq_len=96)

分别处理意味着：
- 对16×5=80个独立的时间序列分别进行小波变换
- 开盘价和交易量等不同类型特征可能有完全不同的频率特性
- 不同股票的同一特征(如开盘价)也可能有不同的趋势和波动模式

这种分别处理的方法确保了我们能够精确捕获每个特征序列的长期趋势和短期波动，为后续的模式组装和特征融合提供更准确的输入。

# 时间模式组装

时间模式组装(`TemporalPatternAssembling`)是MSCNN模型中的关键组件，主要负责将解耦后的长期和短期时间模式进行有效融合。下面我将详细解释这个模块的实现原理和工作流程：

### 时间模式组装的核心思想

这个模块实现了一种双向交互机制，通过"全局-局部"和"局部-全局"两个方向的信息流动，增强长期模式和短期模式之间的相互作用，从而生成更有代表性的时间特征。

### 详细实现步骤

#### 1. 模块初始化

```python
def __init__(self, channels, seq_len):
    super(TemporalPatternAssembling, self).__init__()
    
    # 全局到局部映射（用于增强短期模式）
    self.global_to_local = nn.Sequential(
        nn.Conv1d(channels, channels, kernel_size=1),
        nn.BatchNorm1d(channels),
        nn.ReLU(inplace=True)
    )
    
    # 局部到全局映射（用于增强长期模式）
    self.local_to_global = nn.Sequential(
        nn.AdaptiveAvgPool1d(1),  # 全局池化获取全局上下文
        nn.Conv1d(channels, channels, kernel_size=1),
        nn.Sigmoid()  # 生成注意力权重
    )
    
    # 特征融合
    self.fusion = nn.Sequential(
        nn.Conv1d(channels * 2, channels, kernel_size=1),
        nn.BatchNorm1d(channels),
        nn.ReLU(inplace=True)
    )
```

这里定义了三个主要组件：
- **全局到局部映射(global_to_local)**: 处理长期模式，输出用于增强短期模式的特征
- **局部到全局映射(local_to_global)**: 处理短期模式，输出用于增强长期模式的特征
- **特征融合(fusion)**: 将增强后的长期和短期模式融合为单一特征表示

#### 2. 前向传播过程

```python
def forward(self, long_term_pattern, short_term_pattern):
    # 全局到局部交互：使用长期模式增强短期模式
    global_context = self.global_to_local(long_term_pattern)
    enhanced_short_term = short_term_pattern * global_context
    
    # 局部到全局交互：使用短期模式增强长期模式
    local_context = self.local_to_global(short_term_pattern)
    enhanced_long_term = long_term_pattern * local_context
    
    # 特征融合
    concat_features = torch.cat([enhanced_long_term, enhanced_short_term], dim=1)
    assembled_features = self.fusion(concat_features)
    
    return assembled_features
```

前向传播分为三个关键步骤：

##### 2.1 全局到局部交互 (Global-to-Local Interaction)

```python
global_context = self.global_to_local(long_term_pattern)
enhanced_short_term = short_term_pattern * global_context
```

- **输入**: 长期模式(低频分量) `[batch_size, channels, seq_len]`
- **处理**: 通过1×1卷积、批归一化和ReLU激活转换长期模式信息
- **增强**: 将处理后的长期模式信息与短期模式相乘，实现点乘注意力机制
- **目的**: 使短期波动特征受到长期趋势的引导和约束
- **直观理解**: 让短期波动在长期趋势的"大背景"下被重新理解和权重化

##### 2.2 局部到全局交互 (Local-to-Global Interaction)

```python
local_context = self.local_to_global(short_term_pattern)
enhanced_long_term = long_term_pattern * local_context
```

- **输入**: 短期模式(高频分量) `[batch_size, channels, seq_len]`
- **处理**: 
  1. 首先通过`AdaptiveAvgPool1d(1)`将时间维度压缩为1，提取全局上下文
  2. 然后通过1×1卷积处理
  3. 最后通过Sigmoid函数生成0-1范围的注意力权重
- **增强**: 将生成的注意力权重与长期模式相乘
- **目的**: 让长期趋势特征关注短期波动中重要的部分
- **直观理解**: 短期波动提供"关注提示"，引导长期趋势对某些时间点或特征通道更加关注

##### 2.3 特征融合 (Feature Fusion)

```python
concat_features = torch.cat([enhanced_long_term, enhanced_short_term], dim=1)
assembled_features = self.fusion(concat_features)
```

- **输入**: 增强后的长期和短期模式 `[batch_size, channels*2, seq_len]`
- **处理**: 
  1. 在通道维度上拼接两种增强模式
  2. 通过1×1卷积将通道数从`channels*2`降回`channels`
  3. 批归一化和ReLU激活
- **输出**: 融合后的特征 `[batch_size, channels, seq_len]`
- **目的**: 综合利用长期趋势和短期波动的信息，生成更全面的特征表示

### 技术价值和意义

1. **双向信息流**: 不仅长期模式影响短期模式，短期模式也反过来影响长期模式，形成双向交互
   
2. **注意力机制**: 使用乘法注意力实现特征增强，关注重要的时间模式，抑制不重要的部分

3. **信息互补**: 长期模式和短期模式各自捕获不同频率的特征，通过交互互相补充信息

4. **维度统一**: 最终输出与原始输入保持相同的通道数，便于后续处理和残差连接

5. **非线性表达**: 多次使用非线性激活函数(ReLU和Sigmoid)增强模型表达能力

### 实际应用示例

以金融时间序列为例，该模块的工作流程可以理解为：

1. **长期趋势指导短期波动理解**: 
   - 在牛市趋势中，短期回调可能被识别为买入机会
   - 在熊市趋势中，短期反弹可能被识别为卖出机会

2. **短期波动影响长期趋势判断**: 
   - 短期波动的强度和频率可能暗示趋势即将变化
   - 短期波动模式可能包含长期趋势未捕获的重要信号

通过这种双向交互和融合，时间模式组装模块能够生成既包含长期趋势信息又融入短期波动特征的综合表示，显著提高模型对时间序列的理解和预测能力。
