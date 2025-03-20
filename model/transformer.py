import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from model.rope import RotaryEmbedding, precompute_freqs_cis, NTKScaledRotaryEmbedding
from model.attention import MultiLatentAttention, MixedLatentAttention
from model.moe import MixtureOfExperts, MLP
from log.logger import get_logger

logger = get_logger(__file__, "transformer.log")

# 添加最新的Dynamic tanh 代替 RMSNorm
"""
DyT（Dynamic Tanh）
基本原理：
    DyT 是一种基于 tanh 函数的动态缩放方法。
    它通过一个可学习的缩放参数 alpha 和 tanh 函数对输入进行非线性变换。
计算步骤：
    计算 tanh(alpha * x)，其中 alpha 是可学习参数。
    进行仿射变换：output = tanh_output * weight + bias。
计算复杂性：
    tanh 函数的计算复杂度较低，因为它是一个简单的逐元素非线性变换。
    由于不需要计算均值或方差，DyT 的计算量主要集中在 tanh 和简单的乘加运算上。
RMSNorm（Root Mean Square Normalization）
基本原理：
    RMSNorm 是一种基于均方根的归一化方法。
    它通过计算输入的均方根（RMS）来进行归一化。
计算步骤：
    计算均方根：rms = sqrt(mean(x^2) + epsilon)。
    进行归一化：output = x / rms * weight。
计算复杂性：
    需要计算输入的平方和均值，这涉及到对整个输入张量的归约操作。
    归约操作（如求均值）通常比逐元素操作（如 tanh）计算量更大，尤其是在大批量或长序列的情况下。
比较与总结
计算量：
    DyT 的计算量主要来自于逐元素的 tanh 变换和简单的仿射变换。
    RMSNorm 需要计算均方根，这涉及到额外的归约操作，通常计算量更大。
应用场景：
    DyT 适用于需要简单非线性变换的场景，计算效率高。
    RMSNorm 提供了一种更稳定的归一化方法，适用于需要对输入进行标准化的场景。
选择依据：
    如果计算效率是主要考虑因素，且不需要严格的归一化，DyT 可能是更好的选择。
    如果需要更稳定的归一化效果，RMSNorm 可能更合适，尽管计算量稍大。
    通过这种比较，我们可以看到，DyT 由于其简单的计算步骤，通常在计算量上比 RMSNorm 更少。
"""
class DyT(nn.Module):
    def __init__(self, num_features, alpha_init=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)  # 可学习的缩放参数
        self.weight = nn.Parameter(torch.ones(num_features))   # 通道缩放参数γ
        self.bias = nn.Parameter(torch.zeros(num_features))     # 通道偏移参数β

    def forward(self, x):
        x = torch.tanh(self.alpha * x)         # 动态缩放并应用tanh
        return x * self.weight + self.bias     # 仿射变换


class RMSNorm(nn.Module):
    """
    均方根层归一化，比LayerNorm更高效
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        初始化RMSNorm

        参数:
            hidden_size: 隐藏层维度
            eps: 数值稳定性参数
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 计算均方根
        norm = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        # 归一化并缩放
        return x / norm * self.weight


class TransformerBlock(nn.Module):
    """
    Transformer编码器块，包含自注意力和前馈网络
    """
    def __init__(
        self, 
        hidden_size: int = 256,
        num_heads: int = 4,
        qk_nope_head_dim: int = 32,
        qk_rope_head_dim: int = 32,
        v_head_dim: int = 64,
        moe_intermediate_size: int = 256,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        q_lora_rank: int = 16,
        kv_lora_rank: int = 32,
        attention_scale_factor: float = 1.0,
        attention_type: str = "mixed",
        max_batch_size: int = 32,
        max_seq_len: int = 128,
        dtype: str = "float16"
    ):
        """
        初始化Transformer编码器块

        参数:
            hidden_size: 隐藏层维度
            num_heads: 注意力头数量
            qk_nope_head_dim: 不使用旋转位置编码的Q/K头维度
            qk_rope_head_dim: 使用旋转位置编码的Q/K头维度
            v_head_dim: 值向量的头维度
            moe_intermediate_size: MoE中间层维度
            num_experts: 专家数量
            num_experts_per_token: 每个token使用的专家数量
            attention_dropout: 注意力Dropout比率
            hidden_dropout: 隐藏层Dropout比率
            q_lora_rank: Q的低秩适应维度
            kv_lora_rank: K和V的低秩适应维度
            attention_scale_factor: 注意力缩放因子
            attention_type: 注意力类型,可选mixed,stardard,gqa
            max_batch_size: 最大批量大小
            max_seq_len: 最大序列长度
        """
        super().__init__()
        
        # 保存参数
        self.hidden_size = hidden_size
        self.attention_type = attention_type
        
        # 前注意力层归一化
        self.pre_attn_norm  = RMSNorm(hidden_size)
        # 打印pre_attn_norm的权重shape
        logger.info(f"3. pre_attn_norm的权重shape: {self.pre_attn_norm.weight.shape}")
        # 注意力机制
        if self.attention_type == "mixed":
            self.attention = MixedLatentAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                dropout=attention_dropout,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                attention_scale_factor=attention_scale_factor,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                dtype=dtype
            )
        else:
            self.attention = MultiLatentAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                dropout=attention_dropout,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                attention_scale_factor=attention_scale_factor
            )
        # 打印attention的权重shape
        logger.info(f"4. attention的权重shape: {self.attention.weight.shape}")
        
        # 前MoE层归一化
        self.pre_moe_norm = RMSNorm(hidden_size)
        # 打印pre_moe_norm的权重shape
        logger.info(f"5. pre_moe_norm的权重shape: {self.pre_moe_norm.weight.shape}")
        
        # MoE前馈网络
        self.moe = MixtureOfExperts(
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            num_shared_experts=1,
            shared_intermediate_size=moe_intermediate_size * 4
        )
        
        # Dropout层
        self.dropout = nn.Dropout(hidden_dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播

        参数:
            hidden_states: 输入张量 [batch_size, seq_len, hidden_size]
            freqs_cis: 旋转位置编码的频率张量
            attention_mask: 注意力掩码
            position_ids: 位置ID
            start_pos: 起始位置（用于增量解码）
            output_attentions: 是否输出注意力权重

        返回:
            (output, attention_weights) 元组
        """
        # 残差连接的输入
        residual = hidden_states
        
        # 1. 自注意力层
        # 层归一化
        hidden_states = self.pre_attn_norm(hidden_states)
        
        # 应用注意力
        if self.attention_type == "mixed":
            attn_output = self.attention(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                position_ids=position_ids,
                start_pos=start_pos,
                output_attentions=output_attentions
            )
            
            # 处理返回值
            if output_attentions:
                hidden_states, attention_weights = attn_output
            else:
                hidden_states = attn_output
                attention_weights = None
        else:
            attn_outputs = self.attention(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions
            )
            
            hidden_states = attn_outputs[0]
            attention_weights = attn_outputs[2] if output_attentions else None
        
        # 应用dropout
        hidden_states = self.dropout(hidden_states)
        
        # 残差连接
        hidden_states = residual + hidden_states
        
        # 2. MoE前馈网络
        # 保存残差连接的输入
        residual = hidden_states
        
        # 层归一化
        hidden_states = self.pre_moe_norm(hidden_states)
        
        # 应用MoE
        hidden_states = self.moe(hidden_states)
        
        # 应用dropout
        hidden_states = self.dropout(hidden_states)
        
        # 残差连接
        hidden_states = residual + hidden_states
        
        return (hidden_states, attention_weights) if output_attentions else (hidden_states, None)


class StockTransformerModel(nn.Module):
    """
    股票价格预测的Transformer模型
    """
    def __init__(
        self,
        vocab_size: int = 64,  # 对于数值特征，这是特征维度
        hidden_size: int = 256,
        dtype: str = "float16",
        num_layers: int = 4,
        num_heads: int = 4,
        qk_nope_head_dim: int = 32,
        qk_rope_head_dim: int = 32,
        v_head_dim: int = 64,
        moe_intermediate_size: int = 256,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        max_seq_len: int = 128,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        q_lora_rank: int = 16,
        kv_lora_rank: int = 32,  
        attention_scale_factor: float = 1.0,
        attention_type: str = "mixed",
        max_batch_size: int = 32,
        rope_scaling_factor: float = 1.0,
        rope_theta: float = 10000.0,
        prediction_type: str = "regression",
        rotary_type: str = "rope"
    ):
        """
        初始化股票价格预测的Transformer模型

        参数:
            vocab_size: 词汇表大小或特征维度
            hidden_size: 隐藏层维度
            num_layers: Transformer层数
            num_heads: 注意力头数量
            qk_nope_head_dim: 不使用旋转位置编码的Q/K头维度
            qk_rope_head_dim: 使用旋转位置编码的Q/K头维度
            v_head_dim: 值向量的头维度
            moe_intermediate_size: MoE中间层维度
            num_experts: 专家数量
            num_experts_per_token: 每个token使用的专家数量
            max_seq_len: 最大序列长度
            attention_dropout: 注意力Dropout比率
            hidden_dropout: 隐藏层Dropout比率
            q_lora_rank: Q的低秩适应维度
            kv_lora_rank: K和V的低秩适应维度
            attention_scale_factor: 注意力缩放因子
            attention_type: 注意力类型,可选mixed,stardard,gqa
            max_batch_size: 最大批量大小
            rope_scaling_factor: 位置编码缩放因子
            rope_theta: 位置编码base
            prediction_type: 预测类型，"regression"或"classification"
            rotary_type: 位置编码类型，可选rope,ntk_rope
        """
        super().__init__()
        
        # 保存参数
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.prediction_type = prediction_type
        
        # 初始化输入投影层（将64维特征投影到256维）
        self.input_projection = nn.Linear(vocab_size, hidden_size)
        logger.info(f"1. input_projection的权重shape: {self.input_projection.weight.shape}")
        
        # 初始化RopE位置编码，位置编码维度信息： rotary_emb -> Tensor[max_seq_len, dim]
        if rotary_type == "ntk_rope":
            self.rotary_emb = NTKScaledRotaryEmbedding(
                dim=qk_rope_head_dim,
                max_seq_len=max_seq_len,
                original_max_len=max_seq_len,
                scaling_factor=rope_scaling_factor,
                beta_fast=32,
                beta_slow=1
            )
        else:
            self.rotary_emb = RotaryEmbedding(
                # RoPE使用的维度需要是偶数，因为RoPE的实现方式需要将数据映射到复数空间
                dim=qk_rope_head_dim,
                # 最大序列长度  
                max_seq_len=max_seq_len,
                # 位置编码base
                theta=rope_theta,
                # 位置编码缩放因子
                scaling_factor=rope_scaling_factor
            )
        # 打印rotary_emb的权重shape
        logger.info(f"2. rotary_emb的权重shape: {self.rotary_emb.freqs_buffer.shape}")

        # Transformer层 (num_layers=4)
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                moe_intermediate_size=moe_intermediate_size,
                num_experts=num_experts,
                num_experts_per_token=num_experts_per_token,
                attention_dropout=attention_dropout,
                hidden_dropout=hidden_dropout,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                attention_scale_factor=attention_scale_factor,
                attention_type=attention_type,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                dtype=dtype
            ) for _ in range(num_layers)
        ])
        
        # 最终层归一化
        self.final_norm = RMSNorm(hidden_size)
        
        # 预测头
        if prediction_type == "regression":
            # 回归: 预测连续值
            self.head = nn.Linear(hidden_size, 1)
        else:
            # 分类: 预测上涨/下跌/不变
            self.head = nn.Linear(hidden_size, 3)
        
        # 初始化子模块所有的模型权重__init_weights()
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
    
    def get_input_embeddings(self):
        """获取输入嵌入层"""
        return self.input_projection
    
    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        参数:
            inputs: 输入张量 [batch_size, seq_len, feature_dim]
            attention_mask: 注意力掩码
            position_ids: 位置ID
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出隐藏状态

        返回:
            包含模型输出的字典
        """
        batch_size, seq_len, _ = inputs.shape
        
        # 生成旋转位置编码
        freqs_cis = self.rotary_emb.freqs_buffer[:seq_len]
        max_len = freqs_cis.size(0)
        
        # 如果没有提供位置ID，创建默认位置ID
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=inputs.device) % max_len
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        else:
            # 确保提供的position_ids不超过最大长度
            position_ids = position_ids % max_len
        
        # 输入投影到隐藏维度
        hidden_states = self.input_projection(inputs)
        
        # 存储所有层的隐藏状态
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # 通过所有Transformer层
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            layer_outputs = layer(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions and layer_outputs[1] is not None:
                all_attentions += (layer_outputs[1],)
        
        # 最终层归一化
        hidden_states = self.final_norm(hidden_states)
        
        # 添加最后一层输出到隐藏状态
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        # 只使用序列的最后一个位置进行预测
        # 这适合股价预测任务，我们通常只关心最后一个时间步的预测
        last_hidden_state = hidden_states[:, -1, :]
        
        # 应用预测头
        logits = self.head(last_hidden_state)
        
        # 根据预测类型处理输出
        if self.prediction_type == "regression":
            prediction = logits.squeeze(-1)  # [batch_size]
        else:
            prediction = F.softmax(logits, dim=-1)  # [batch_size, 3]
        
        # 构建输出字典
        outputs = {
            "prediction": prediction,
            "last_hidden_state": hidden_states
        }
        
        if output_hidden_states:
            outputs["hidden_states"] = all_hidden_states
            
        if output_attentions:
            outputs["attentions"] = all_attentions
        
        return outputs


class StockPricePredictor:
    """
    股票价格预测器，包装Transformer模型并提供易用的接口
    """
    def __init__(
        self,
        feature_dim: int,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        qk_nope_head_dim: int = 32,
        qk_rope_head_dim: int = 32,
        v_head_dim: int = 64,
        max_seq_len: int = 60,
        prediction_type: str = "regression",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化股票价格预测器

        参数:
            feature_dim: 特征维度
            hidden_size: 隐藏层维度
            num_layers: Transformer层数
            num_heads: 注意力头数量
            qk_nope_head_dim: 不使用旋转位置编码的Q/K头维度
            qk_rope_head_dim: 使用旋转位置编码的Q/K头维度
            v_head_dim: 值向量的头维度
            max_seq_len: 最大序列长度
            prediction_type: 预测类型，"regression"或"classification"
            device: 运行设备
        """
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.prediction_type = prediction_type
        self.device = device
        
        # 创建模型
        self.model = StockTransformerModel(
            vocab_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            max_seq_len=max_seq_len,
            prediction_type=prediction_type
        ).to(device)
    
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        使用模型进行预测

        参数:
            features: 特征张量 [batch_size, seq_len, feature_dim]

        返回:
            预测结果
        """
        # 确保输入在正确的设备上
        features = features.to(self.device)
        
        # 设置为评估模式
        self.model.eval()
        
        # 禁用梯度计算以加速推理
        with torch.no_grad():
            outputs = self.model(features)
            
        return outputs["prediction"]
    
    def save(self, path: str):
        """
        保存模型

        参数:
            path: 保存路径
        """
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "feature_dim": self.feature_dim,
            "hidden_size": self.hidden_size,
            "max_seq_len": self.max_seq_len,
            "prediction_type": self.prediction_type
        }, path)
        
    @classmethod
    def load(cls, path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        加载模型

        参数:
            path: 模型路径
            device: 运行设备

        返回:
            加载的模型
        """
        checkpoint = torch.load(path, map_location=device)
        
        # 创建预测器实例
        predictor = cls(
            feature_dim=checkpoint["feature_dim"],
            hidden_size=checkpoint["hidden_size"],
            max_seq_len=checkpoint["max_seq_len"],
            prediction_type=checkpoint["prediction_type"],
            device=device
        )
        
        # 加载模型权重
        predictor.model.load_state_dict(checkpoint["model_state_dict"])
        
        return predictor


# 测试代码
if __name__ == "__main__":
    # 测试参数
    batch_size = 16
    seq_len = 128
    feature_dim = 64  # 股票数据的特征维度
    hidden_size = 256
    num_layers = 4
    num_heads = 4
    
    # 创建测试输入
    inputs = torch.randn(batch_size, seq_len, feature_dim)
    
    # 测试Transformer模型
    logger.info(f"测试股票Transformer模型...")
    model = StockTransformerModel(
        vocab_size=feature_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=seq_len,
        prediction_type="regression"
    )
    
    outputs = model(inputs)
    
    logger.info(f"输入形状: {inputs.shape}")
    logger.info(f"预测形状: {outputs['prediction'].shape}")
    logger.info(f"最后隐藏状态形状: {outputs['last_hidden_state'].shape}")
    
    # 测试预测器
    logger.info(f"测试股票价格预测器...")
    predictor = StockPricePredictor(
        feature_dim=feature_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=seq_len,
        prediction_type="regression",
        device="cpu"
    )
    
    predictions = predictor.predict(inputs)
    
    logger.info(f"输入形状: {inputs.shape}")
    logger.info(f"预测形状: {predictions.shape}")
    
    logger.info(f"所有测试通过！") 