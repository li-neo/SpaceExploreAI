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
