from dataclasses import dataclass
from typing import Optional, List, Literal


@dataclass
class ModelConfig:
    """SpaceExploreAI Configs"""
    
    # 模型架构参数
    model_name: str = "SpaceExploreAI"
    model_type: str = "transformer"  # transformer, moe
    vocab_size: int = 1024  # 词汇表大小
    hidden_size: int = 1024  # 隐藏层大小
    num_hidden_layers: int = 16  # 隐藏层数量
    num_attention_heads: int = 4  # 注意力头数量
    intermediate_size: int = 1024  # 中间层大小
    hidden_act: str = "silu"  # 激活函数swiGLU 贝塔=1
    max_position_embeddings: int = 1024  # 最大位置编码
    
    # RoPE参数
    rope_theta: float = 10000.0
    rope_scaling: dict = None  # 默认为None，可以设置为{"type": "linear", "factor": 4.0}
    
    # MLA (多头潜在注意力) 参数
    qk_nope_head_dim: int = 64  # 非位置编码的query/key维度
    qk_rope_head_dim: int = 32  # 使用旋转位置编码的query/key维度
    v_head_dim: int = 64  # value维度
    
    # MOE参数
    moe_enabled: bool = True
    num_experts: int = 16  # 专家数量
    num_experts_per_token: int = 4  # 每个token使用的专家数量
    
    # 训练参数
    batch_size: int = 32
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    num_train_epochs: int = 3
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    
    # 数据参数
    train_data_path: str = "data/processed/train"
    eval_data_path: str = "data/processed/eval"
    test_data_path: str = "data/processed/test"
    sequence_length: int = 1024
    
    # 股票特定参数
    time_features: List[str] = None  # 时间特征
    price_features: List[str] = None  # 价格特征
    technical_indicators: List[str] = None  # 技术指标
    fundamental_features: List[str] = None  # 基本面特征
    sentiment_features: List[str] = None  #  情绪指标
    relative_features: List[str] = None  # 相对大盘强弱指标
    macro_features: List[str] = None  # 宏观指标
    news_features: List[str] = None  # 新闻指标
    
    # 推理参数
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 5
    
    # 分布式训练参数
    use_deepspeed: bool = True
    deepspeed_stage: int = 3  # ZeRO阶段
    use_flash_attention: bool = True
    
    def __post_init__(self):
        if self.time_features is None:
            self.time_features = ["day_of_week", "month", "year", "day_of_month"]
        
        if self.price_features is None:
            self.price_features = ["open", "high", "low", "close", "volume", "adj_close"]
        
        if self.technical_indicators is None:
            self.technical_indicators = [
                "ma_5", "ma_10", "ma_20", "ma_50", "ma_200",  # 移动平均线
                "rsi_14", "macd", "macd_signal", "macd_hist",  # 动量指标
                "bb_upper", "bb_middle", "bb_lower",  # 布林带
                "atr_14",  # 平均真实范围
                "obv",  # 能量指标
                "stoch_k", "stoch_d"  # 随机指标
            ]
        
        if self.fundamental_features is None:
            self.fundamental_features = [
                "pe_ratio", "pb_ratio", "dividend_yield", "market_cap", 
                "revenue_growth", "profit_margin", "debt_to_equity"
            ]
        
        if self.sentiment_features is None:
            self.sentiment_features = [
                "news_sentiment", "social_media_sentiment", 
                "analyst_recommendations", "institutional_ownership_change"
            ]
            
        if self.rope_scaling is None and self.max_position_embeddings > 1024:
            self.rope_scaling = {"type": "linear", "factor": self.max_position_embeddings / 1024}


@dataclass
class DeepSpeedConfig:
    """DeepSpeed配置类"""
    
    zero_optimization: dict = None
    optimizer: dict = None
    scheduler: dict = None
    gradient_clipping: float = 1.0
    fp16: dict = None
    bf16: dict = None
    
    def __post_init__(self):
        if self.zero_optimization is None:
            self.zero_optimization = {
                "stage": 3,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e7,
                "allgather_bucket_size": 5e7
            }
            
        if self.optimizer is None:
            self.optimizer = {
                "type": "AdamW",
                "params": {
                    "lr": 1e-4,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": 0.1
                }
            }
            
        if self.scheduler is None:
            self.scheduler = {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 1e-4,
                    "warmup_num_steps": 2000,
                    "total_num_steps": 50000
                }
            }
            
        if self.fp16 is None and self.bf16 is None:
            self.bf16 = {"enabled": True}


# 默认配置
default_model_config = ModelConfig()
default_deepspeed_config = DeepSpeedConfig() 