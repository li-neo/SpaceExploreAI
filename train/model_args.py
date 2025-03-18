from typing import List, Literal
from dataclasses import dataclass
@dataclass
class ModelArgs():
    """ Model Args - 模型参数配置类

    1. 注意力机制参数
        hidden_size = 256
        num_heads = 4
        qk_nope_head_dim = 32
        qk_rope_head_dim = 32
        v_head_dim = 64
        每个注意力头的参数:
        Q投影矩阵(不使用旋转位置编码):256 * 32 = 8,192 参数
        K投影矩阵(不使用旋转位置编码):256 * 32 = 8,192 参数
        Q投影矩阵(使用旋转位置编码):256 * 32 = 8,192 参数
        K投影矩阵(使用旋转位置编码):256 * 32 = 8,192 参数
        V投影矩阵:256 * 64 = 16,384 参数
        输出投影矩阵:
        输出投影:(32 + 32 + 64) * 256 = 128 * 256 = 32,768 参数
        每个注意力层的总参数:8,192 + 8,192 + 8,192 + 8,192 + 16,384 + 32,768 = 81,920 参数
    2. MoE (Mixture of Experts) 参数
        hidden_size = 256
        moe_intermediate_size = 256
        num_experts = 8
        num_experts_per_token = 2
        每个专家包含两个线性层:
        上投影:256 * 256 = 65,536 参数
        下投影:256 * 256 = 65,536 参数
        所有专家的参数:8 * (65,536 + 65,536) = 8 * 131,072 = 1,048,576 参数
        路由网络参数:
        路由器:256 * 8 = 2,048 参数
    3. 层归一化参数
        每个Transformer层的层归一化:
        每个层归一化:2 * 256 = 512 参数
        每层的层归一化总参数:2 * 512 = 1,024 参数
    4. 总参数计算
        每个Transformer层的参数:
        注意力机制:81,920 参数
        MoE:1,048,576 + 2,048 = 1,050,624 参数
        层归一化:1,024 参数
        每层总计:81,920 + 1,050,624 + 1,024 = 1,133,568 参数
        所有Transformer层的参数:
        4 * 1,133,568 = 4,534,272 参数
    5. 输入和输出层参数
        假设输入特征维度为10,输出维度为1:
        输入嵌入层:10 * 256 = 2,560 参数
        输出层:256 * 1 = 256 参数
    6. 总参数量
        模型总参数量:
        Transformer层:4,534,272 参数
        输入嵌入层:2,560 参数
        输出层:256 参数
        总计:4,534,272 + 2,560 + 256 = 4,537,088 参数 ≈ 454万参数

    """
    # 数据相关参数
    feature_dim: int = 64  # 特征维度,输入特征的维度大小, 对应数据的数值列64个列
    raw_data_dir: str = "data/raw"  # 原始数据目录路径
    processed_data_dir: str = "data/processed"  # 处理后数据存储目录路径
    tickers: str = "TSLA"  # 股票代码,多个股票代码用逗号分隔,如"TSLA,AAPL,GOOG"
    data_source: str = "yahoo"  # 数据源,可选"yahoo"或"alphavantage"等
    load_processed: bool = True  # 是否加载已处理的数据,True表示直接加载处理好的数据
    merge_stocks: bool = False  # 是否合并多只股票的数据,True表示将多只股票数据合并为一个数据集
    scaler_type: str = "robust"  # 数据缩放器类型,可选"standard"(标准化)、"minmax"(最小最大化)或"robust"(稳健缩放)
    test_size: float = 0.1  # 测试集比例,占总数据的10%
    val_size: float = 0.1  # 验证集比例,占总数据的10%
    sequence_length: int = 32  # 输入序列长度,即每个样本包含的时间步数,较小的值可降低内存使用,应该与数据处理的seq保持一致 DataArgs.seq_length
    max_sequence_length: int = 128  # 模型能处理的最大序列长度,用于定义模型架构的上限
    prediction_horizon: int = 2  # 预测周期,即预测未来多少个时间步
    feature_groups: str = None  # 特征组,指定使用哪些特征组,多个用逗号分隔
    batch_size: int = 16  # 训练批量大小,每次更新使用的样本数量
    max_batch_size: int = 32  # 最大批量大小,用于推理或特殊情况
    num_workers: int = 2  # 数据加载线程数,用于并行数据加载,较小的值适合Mac的CPU核心数
    
    # 模型相关参数 参数共计500M
    hidden_size: int = 256  # 隐藏层维度,模型内部表示的维度大小
    num_layers: int = 4  # Transformer层数,堆叠的Transformer编码器层数
    num_heads: int = 4  # 注意力头数量,多头注意力机制中的头数
    moe_intermediate_size: int = 256  # MoE中间层维度,混合专家模型中间层的维度
    num_experts: int = 8  # 专家数量,混合专家模型中的专家数量
    num_experts_per_token: int = 2  # 每个token使用的专家数量,每个输入会激活的专家数量

    # mla
    # 混合注意力
    attention_type: str = "mixed"  # 注意力类型,可选mixed,stardard,gqa
    attention_dropout: float = 0.1  # 注意力Dropout比率,注意力机制中的丢弃率
    hidden_dropout: float = 0.1  # 隐藏层Dropout比率,前馈网络中的丢弃率
    attention_scale_factor: float = 1.0  # 注意力缩放因子,用于调整注意力机制的权重
   

    # 低秩适应
    q_lora_rank: int = 0  # 低秩适应的Q矩阵的秩, 0表示不使用低秩适应
    kv_lora_rank: int = 32  # 低秩适应的K/V矩阵的秩,建议设置为hidden_size的 1/4 - 1/8
    # qk_head_dim = qk_nope_head_dim + qk_rope_head_dim 
    qk_nope_head_dim: int = 32  # 不使用旋转位置编码的Q/K头维度,每个注意力头中查询和键向量的维度
    qk_rope_head_dim: int = 32  # 使用旋转位置编码的Q/K头维度,使用RoPE的注意力头中查询和键向量的维度
    v_head_dim: int = 64  # 值向量的头维度,每个注意力头中值向量的维度
    
    # 位置编码相关参数
    # TODO:建议改成1000，因为我们vocab_size=64
    rope_theta: float = 10000.0  # RoPE的基频参数,控制旋转位置编码的频率分布
    rope_scaling_factor: float = 1.0  # RoPE缩放因子,用于调整位置编码的尺度
    rope_scaling_type: str = "linear"  # RoPE缩放类型,可选"linear"(线性)或"ntk"(NTK)
    # 数据类型和精度
    dtype: Literal["float32", "float16", "bfloat16"] = "float16"  # 模型权重的数据类型,float32为标准精度,float16和bfloat16为低精度
    
    # 训练相关参数
    learning_rate: float = 8e-5  # 学习率,控制参数更新的步长大小
    weight_decay: float = 0.01  # 权重衰减,L2正则化系数,用于防止过拟合
    clip_grad_norm: float = 1.0  # 梯度裁剪范数,限制梯度的最大范数,防止梯度爆炸
    num_epochs: int = 16  # 训练轮次,完整遍历训练集的次数
    patience: int = 3  # 早停耐心,验证集性能不再提升的轮次数,超过此值则停止训练
    save_dir: str = "models"  # 模型保存目录,训练好的模型权重保存位置
    model_name: str = "SpaceExploreAI"  # 模型名称,保存模型时使用的名称前缀
    log_interval: int = 5  # 日志记录间隔,每训练多少批次记录一次日志
    device: str = "mps"  # 训练设备,"mps"表示使用Mac的Metal Performance Shaders进行GPU加速
    disable_mixed_precision: bool = True  # 是否禁用混合精度训练,True表示禁用,在Mac上通常需要禁用
    resume_from: str = None  # 从检查点恢复训练,指定检查点文件路径
    seed: int = 42  # 随机种子,确保实验可重复性的随机数种子

    prediction_type: str = "regression"  # 预测类型,"regression"(回归)或"classification"(分类)
