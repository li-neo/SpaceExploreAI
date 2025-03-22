"""
# 此python文件用于生成HuggingFace开源文件、包括:
├── pytorch_model.bin
├── tokenizer.json
├── special_tokens_map.json
├── tokenizer_config.json
├── vocab.txt
├── config.json
├── README.md
"""

import os
import json
import shutil
import torch
import argparse
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, PretrainedConfig

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 导入ModelArgs和数据处理相关模块
from train.model_args import ModelArgs
from data.data_processor import StockDataProcessor
from data.technical_indicators import TechnicalIndicatorProcessor

# 定义目标目录
TARGET_DIR = "/Users/li/AIProject/HFSpaceExploreAI/SpaceExploreAI-Small-Base-Regression-5M"

def ensure_dir_exists(directory):
    """确保目录存在，如果不存在则创建"""
    os.makedirs(directory, exist_ok=True)
    print(f"确保目录存在: {directory}")

def mv_model_file(source_model_path, target_dir):
    """移动模型文件到目标目录"""
    target_path = os.path.join(target_dir, os.path.basename(source_model_path))
    shutil.move(source_model_path, target_path)
    print(f"移动模型文件: {source_model_path} -> {target_path}")

def copy_model_file(source_model_path, target_dir):
    """复制模型文件到目标目录并重命名为pytorch_model.bin"""
    target_path = os.path.join(target_dir, "pytorch_model.bin")
    
    # 如果源文件是pt格式，先加载并重新保存为pytorch_model.bin
    if source_model_path.endswith('.pt'):
        print(f"加载模型: {source_model_path}")
        model = torch.load(source_model_path, map_location=torch.device('cpu'))
        
        # 根据模型类型提取需要的状态字典
        if isinstance(model, dict) and 'model_state_dict' in model:
            # 模型是以字典形式保存的
            state_dict = model['model_state_dict']
        elif hasattr(model, 'state_dict'):
            # 模型是模型对象
            state_dict = model.state_dict()
        else:
            # 假设模型直接是状态字典
            state_dict = model
            
        print(f"保存模型到: {target_path}")
        torch.save(state_dict, target_path)
    else:
        # 直接复制文件
        print(f"复制模型文件: {source_model_path} -> {target_path}")
        shutil.copy2(source_model_path, target_path)
    
    return target_path

def create_model_config(target_dir, model_args=None, additional_config=None):
    """创建模型配置文件 config.json
    
    Args:
        target_dir (str): 目标目录路径
        model_args (object, optional): 包含模型参数的对象
        additional_config (dict, optional): 额外的配置项，将覆盖或添加到最终配置中
        
    Returns:
        str: 配置文件路径
    """
    print("创建模型配置文件...")
    
    # 基础配置
    config = {
        "model_type": "space_explore_ai_financial",
        "architectures": ["SpaceExploreAIFinancialModel"],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "gelu",
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-12,
        "pad_token_id": 0,
        "unk_token_id": 3,
        "transformers_version": "4.36.0",
        "use_cache": True,
        "vocab_size": 512,  # 默认值，如果有model_args会被覆盖
        "tokenizer_class": "FinancialFeatureProcessor"
    }
    
    # 如果存在model_args，从中提取配置
    if model_args:
        # 映射ModelArgs属性到config
        mapping = {
            "dim": "hidden_size",
            "n_layers": "num_hidden_layers",
            "n_heads": "num_attention_heads",
            "feature_dim": "vocab_size",
            "multiple_of": "intermediate_multiple",
            "norm_eps": "layer_norm_epsilon",
            "max_seq_len": "max_position_embeddings",
            "context_length": "model_max_length",
            "temp": "temperature",
            "ffn_dim_multiplier": "ffn_dim_multiplier"
        }
        
        # 将ModelArgs中的属性添加到config中
        for attr, config_key in mapping.items():
            if hasattr(model_args, attr):
                value = getattr(model_args, attr)
                config[config_key] = value
        
        # 添加模型名称信息
        if hasattr(model_args, "model_name"):
            config["model_name"] = model_args.model_name
        
        # 添加输出配置
        if hasattr(model_args, "output_type"):
            config["output_type"] = model_args.output_type
            
        # 添加激活函数信息
        if hasattr(model_args, "hidden_act"):
            config["hidden_act"] = model_args.hidden_act
    
    # 添加或覆盖额外的配置项
    if additional_config:
        for key, value in additional_config.items():
            config[key] = value
    
    # 确保必要的参数都有默认值
    required_params = {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "max_position_embeddings": 1024,
        "model_max_length": 1024,
    }
    
    for param, default_value in required_params.items():
        if param not in config:
            config[param] = default_value
            print(f"警告: 配置项 {param} 未提供，使用默认值 {default_value}")
    
    # 保存配置文件
    config_path = os.path.join(target_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"已创建配置文件: {config_path}")
    return config_path

def create_tokenizer_files(target_dir, vocab_size=None, model_args=None):
    """
    创建与 Financial Feature Processor 相关的tokenizer文件
    
    Args:
        target_dir (str): 目标目录路径
        vocab_size (int, optional): 词汇表大小（特征维度）
        model_args (object, optional): 包含模型参数的对象
        
    Returns:
        int: 实际的词汇表大小（特征维度）
    """
    print("创建Financial Feature Processor相关文件...")
    
    # 获取特征信息
    feature_info = get_feature_info(model_args)
    feature_names = feature_info["feature_names"]
    feature_groups = feature_info["feature_groups"]
    
    # 特殊token - 根据项目使用的标准
    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
    
    # 计算实际词汇表大小
    actual_vocab_size = len(special_tokens) + len(feature_names)
    
    # 创建vocab.txt
    print(f"创建vocab.txt，包含{len(special_tokens)}个特殊token和{len(feature_names)}个特征名")
    vocab_path = os.path.join(target_dir, 'vocab.txt')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        # 特殊token
        for token in special_tokens:
            f.write(f"{token}\n")
        # 特征名
        for feature in feature_names:
            f.write(f"{feature}\n")
    
    # 创建tokenizer_config.json
    tokenizer_config = {
        "feature_dim": actual_vocab_size,
        "feature_groups": feature_groups,
        "feature_names": feature_names,
        "do_lower_case": False,
        "model_max_length": 1024,
        "tokenizer_class": "FinancialFeatureProcessor",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>"
    }
    
    # 如果model_args中包含相关配置，更新tokenizer_config
    if model_args:
        tokenizer_attrs = [
            "feature_dim", "model_max_length", "context_length"
        ]
        for attr in tokenizer_attrs:
            if hasattr(model_args, attr):
                value = getattr(model_args, attr)
                if attr == "feature_dim" and value != actual_vocab_size:
                    print(f"警告: model_args中的feature_dim({value})与计算得到的词汇表大小({actual_vocab_size})不一致")
                tokenizer_config[attr] = value
    
    # 保存tokenizer_config.json
    tokenizer_config_path = os.path.join(target_dir, 'tokenizer_config.json')
    with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    
    # 创建special_tokens_map.json
    special_tokens_map = {
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "bos_token": "<bos>",
        "eos_token": "<eos>"
    }
    special_tokens_map_path = os.path.join(target_dir, 'special_tokens_map.json')
    with open(special_tokens_map_path, 'w', encoding='utf-8') as f:
        json.dump(special_tokens_map, f, indent=2, ensure_ascii=False)
    
    # 创建tokenizer.json (更详细的tokenizer配置)
    tokenizer_full = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": i,
                "special": True,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False
            } for i, token in enumerate(special_tokens)
        ] + [
            {
                "id": i + len(special_tokens),
                "special": False,
                "content": feature,
                "single_word": True,
                "lstrip": False,
                "rstrip": False,
                "normalized": False
            } for i, feature in enumerate(feature_names)
        ],
        "normalizer": {
            "type": "FinancialFeatureNormalizer"
        },
        "pre_tokenizer": {
            "type": "FinancialFeaturePreTokenizer"
        },
        "post_processor": {
            "type": "FinancialFeaturePostProcessor"
        },
        "decoder": {
            "type": "FinancialFeatureDecoder"
        },
        "model": {
            "type": "FinancialFeatureProcessor",
            "vocab": {token: i for i, token in enumerate(special_tokens + feature_names)},
            "feature_dim": actual_vocab_size
        }
    }
    tokenizer_path = os.path.join(target_dir, 'tokenizer.json')
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_full, f, indent=2, ensure_ascii=False)
    
    print(f"已创建tokenizer文件: vocab.txt, tokenizer_config.json, special_tokens_map.json, tokenizer.json")
    return actual_vocab_size

def get_feature_info(model_args=None):
    """获取特征信息
    
    Returns:
        字典，包含特征名称列表和特征组
    """
    # 直接使用用户提供的特征列表
    feature_names = [
        'close', 'high', 'low', 'open', 'volume', 
        'day_of_week', 'day_of_month', 'week_of_year', 'month', 'quarter', 'year', 
        'is_month_start', 'is_month_end', 'is_week_start', 'is_week_end', 
        'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5', 'close_lag_10', 
        'high_lag_1', 'high_lag_2', 'high_lag_3', 'high_lag_5', 'high_lag_10', 
        'low_lag_1', 'low_lag_2', 'low_lag_3', 'low_lag_5', 'low_lag_10', 
        'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5', 'volume_lag_10', 
        'future_return_1d', 'future_return_2d', 'future_return_10d', 
        'past_return_1d', 'past_return_2d', 'past_return_5d', 'past_return_10d', 
        'daily_range', 'daily_range_abs', 'gap', 'close_to_open', 'close_to_high', 'close_to_low', 
        'volatility_5d', 'volatility_10d', 'volatility_21d', 
        'atr_5d', 'atr_10d', 'atr_21d', 
        'momentum_5d', 'momentum_10d', 'momentum_21d', 
        'volume_change_5d', 'volume_change_10d', 'volume_change_21d', 
        'close_to_ma_5d', 'close_to_ma_10d', 'close_to_ma_21d', 
        'volume_change_rate'
    ]
    
    # 添加date字段
    if 'date' not in feature_names:
        feature_names.insert(0, 'date')
    
    # 分组特征
    feature_groups = ["price", "time", "lag", "return", "volatility", "volume"]
    
    print(f"使用指定的特征列表，共{len(feature_names)}个特征")
    
    return {
        "feature_names": feature_names,
        "feature_groups": feature_groups
    }

def create_readme(target_dir, model_args=None):
    """创建README.md文件
    
    Args:
        target_dir: 目标目录
        model_args: ModelArgs实例，如果提供则从中读取配置
    """
    # 默认参数
    model_name = "SpaceExploreAI-Small-Base-Regression-5M"
    hidden_size = 512
    num_layers = 4
    num_heads = 4
    sequence_length = 1024
    moe_enabled = False
    moe_details = ""
    attention_type = "混合"
    norm_type = "rmsnorm"
    prediction_type = "regression"
    
    # 如果提供了ModelArgs，则从中读取相关配置
    if model_args:
        model_name = getattr(model_args, "model_name", model_name) + "-Small-Base-Regression-5M"
        hidden_size = getattr(model_args, "hidden_size", hidden_size)
        num_layers = getattr(model_args, "num_layers", num_layers)
        num_heads = getattr(model_args, "num_heads", num_heads)
        sequence_length = getattr(model_args, "sequence_length", sequence_length)
        attention_type = getattr(model_args, "attention_type", attention_type)
        norm_type = getattr(model_args, "norm", norm_type)
        prediction_type = getattr(model_args, "prediction_type", prediction_type)
        
        # MOE 详细信息
        num_experts = getattr(model_args, "num_experts", 0)
        num_experts_per_token = getattr(model_args, "num_experts_per_token", 0)
        moe_enabled = num_experts > 0 and num_experts_per_token > 0
        
        if moe_enabled:
            intermediate_size = getattr(model_args, "moe_intermediate_size", hidden_size * 4)
            moe_details = f"\n- **MOE配置**：{num_experts}个专家，每个token使用{num_experts_per_token}个专家，中间层大小{intermediate_size}"
    
    # 准备技术细节部分
    tech_details = f"""## 技术规格

- **参数量**：约 5M
- **模型类型**：Transformer
- **隐藏层大小**：{hidden_size}
- **隐藏层数量**：{num_layers}
- **注意力头数量**：{num_heads}
- **注意力类型**：{attention_type}
- **归一化类型**：{norm_type}
- **最大序列长度**：{sequence_length}
- **预测类型**：{prediction_type}
- **使用 MoE**：{"是" if moe_enabled else "否"}{"，混合专家模型增强了模型的表达能力" if moe_enabled else "（此版本不使用混合专家模型）"}{moe_details}"""

    # 准备特点部分，根据不同模型类型强调不同的特点
    if prediction_type.lower() == "regression":
        task_description = "回归预测：专为价格预测等回归任务优化"
    else:
        task_description = "分类预测：专为趋势方向预测等分类任务优化"
    
    features_section = f"""### 主要特点

- **轻量化设计**：仅有 5M 参数，适合资源受限环境
- **{task_description}**
- **Transformer 架构**：基于 Transformer 架构，集成了 RoPE 旋转位置编码技术
- **多头{attention_type}注意力**：使用先进的多头注意力机制捕捉时间序列数据模式"""

    readme_content = f"""
# {model_name}

## 模型描述

{model_name} 是一个基于深度学习的金融时序预测模型，专为股票价格趋势分析和预测而设计。这是 SpaceExploreAI 系列的小型版本，具有约 5M 参数，针对{prediction_type}任务进行了优化。

{features_section}

{tech_details}

## 使用示例

```python
from transformers import AutoTokenizer, AutoModel
import torch

# 加载模型和分词器
model_name = "SpaceExploreAI/{model_name}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 准备输入数据 (假设您已经有了处理好的金融数据)
inputs = torch.tensor([[0.1, 0.2, 0.3, ...]])  # 您的金融序列数据

# 进行预测
with torch.no_grad():
    outputs = model(inputs)
    predictions = outputs.last_hidden_state
```

## 免责声明

SpaceExploreAI仅供大模型AI学习、量化交易学习，不可以用于商业用途、不可以以此为投资逻辑，后果自负。

## 许可证

[Apache License 2.0](LICENSE)
"""
    
    readme_path = os.path.join(target_dir, "Usage.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"创建README.md文件: {readme_path}")

def main():
    parser = argparse.ArgumentParser(description='生成HuggingFace模型发布文件')
    parser.add_argument('--model_path', type=str, default='models/SpaceExploreAI_best.pt',
                        help='模型文件路径')
    parser.add_argument('--target_dir', type=str, default=TARGET_DIR,
                        help='生成文件的目标目录')
    parser.add_argument('--vocab_size', type=int, default=None,
                        help='词汇表大小，不指定时从ModelArgs中读取')
    parser.add_argument('--use_model_args', action='store_true', default=True,
                        help='是否使用ModelArgs中的配置')
    parser.add_argument('--data_config', type=str, default='data/data_config.json',
                        help='数据配置文件路径')
    
    args = parser.parse_args()
    
    # 确保目标目录存在
    ensure_dir_exists(args.target_dir)
    
    # 获取ModelArgs
    model_args = None
    if args.use_model_args:
        print("从ModelArgs获取配置...")
        model_args = ModelArgs()
        print(f"成功加载ModelArgs，模型名称: {model_args.model_name}")
    
    # 确定词汇表大小
    vocab_size = args.vocab_size
    if vocab_size is None and model_args:
        vocab_size = model_args.feature_dim
        print(f"使用ModelArgs中的feature_dim作为词汇表大小: {vocab_size}")
    if vocab_size is None:
        # 尝试从数据配置文件读取
        try:
            if os.path.exists(args.data_config):
                with open(args.data_config, 'r') as f:
                    data_config = json.load(f)
                    # 尝试获取特征维度信息
                    feature_groups = data_config.get("feature_groups", [])
                    # 简单估算特征维度大小
                    if feature_groups:
                        # 基本价格特征 + 时间特征 + 基本技术指标 = 大约64维
                        vocab_size = 64
        except Exception as e:
            print(f"读取数据配置文件失败: {e}")
        
        if vocab_size is None:
            vocab_size = 64  # 默认值
        print(f"未明确指定词汇表大小，使用默认值或估算值: {vocab_size}")
    
    # 复制模型文件
    model_path = copy_model_file(args.model_path, args.target_dir)
    
    # 创建tokenizer相关文件先，因为它会返回实际的vocab_size
    actual_vocab_size = create_tokenizer_files(args.target_dir, vocab_size, model_args)
    print(f"通过特征分析得到实际词汇表大小: {actual_vocab_size}")
    
    # 使用实际的vocab_size创建配置文件
    if model_args:
        # 临时更新model_args的feature_dim
        original_feature_dim = getattr(model_args, "feature_dim", None)
        model_args.feature_dim = actual_vocab_size
        
        # 创建配置文件
        create_model_config(args.target_dir, model_args)
        
        # 恢复原始值
        if original_feature_dim is not None:
            model_args.feature_dim = original_feature_dim
    else:
        # 直接使用actual_vocab_size创建配置
        config_data = {"vocab_size": actual_vocab_size}
        create_model_config(args.target_dir, None, config_data)
    
    # 创建README.md
    create_readme(args.target_dir, model_args)

    try:
        # 移动训练曲线图，如果存在的话
        training_curves_path = "models/SpaceExploreAI_training_curves.png"
        if os.path.exists(training_curves_path):
            mv_model_file(training_curves_path, args.target_dir)
            print(f"已移动训练曲线图到目标目录")
    except Exception as e:
        print(f"移动训练曲线图时出错: {e}")
    
    print(f"\n所有文件已成功生成到: {args.target_dir}")
    print("文件列表:")
    for file in os.listdir(args.target_dir):
        file_path = os.path.join(args.target_dir, file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # 转换为MB
        print(f"  - {file} ({file_size:.2f} MB)")

if __name__ == "__main__":
    main()

