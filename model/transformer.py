import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from .rope import RotaryEmbedding, precompute_freqs_cis
from .attention import MultiLatentAttention, MixedLatentAttention
from .moe import MixtureOfExperts, MLP


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
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        moe_intermediate_size: int = 256,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        q_lora_rank: int = 0,
        kv_lora_rank: int = 512,
        attention_scale_factor: float = 1.0,
        use_mixed_attention: bool = True,
        max_batch_size: int = 32,
        max_seq_len: int = 4096
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
            use_mixed_attention: 是否使用混合潜在注意力
            max_batch_size: 最大批量大小
            max_seq_len: 最大序列长度
        """
        super().__init__()
        
        # 保存参数
        self.hidden_size = hidden_size
        self.use_mixed_attention = use_mixed_attention
        
        # 前注意力层归一化
        self.pre_attn_norm = RMSNorm(hidden_size)
        
        # 注意力机制
        if use_mixed_attention:
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
                max_seq_len=max_seq_len
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
        
        # 前MoE层归一化
        self.pre_moe_norm = RMSNorm(hidden_size)
        
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
        if self.use_mixed_attention:
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
        vocab_size: int,  # 对于数值特征，这是特征维度
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        qk_nope_head_dim: int = 32,
        qk_rope_head_dim: int = 32,
        v_head_dim: int = 64,
        moe_intermediate_size: int = 256,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        max_seq_len: int = 4096,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        q_lora_rank: int = 0,
        kv_lora_rank: int = 512,
        attention_scale_factor: float = 1.0,
        use_mixed_attention: bool = True,
        max_batch_size: int = 32,
        scaling_factor: float = 1.0,
        prediction_type: str = "regression"
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
            use_mixed_attention: 是否使用混合潜在注意力
            max_batch_size: 最大批量大小
            scaling_factor: 位置编码缩放因子
            prediction_type: 预测类型，"regression"或"classification"
        """
        super().__init__()
        
        # 保存参数
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.prediction_type = prediction_type
        
        # 输入投影层
        self.input_projection = nn.Linear(vocab_size, hidden_size)
        
        # 位置编码
        self.rotary_emb = RotaryEmbedding(
            dim=qk_rope_head_dim * 2,  # RoPE使用的维度需要是偶数
            max_seq_len=max_seq_len,
            scaling_factor=scaling_factor
        )
        
        # Transformer层
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
                use_mixed_attention=use_mixed_attention,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len
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
        
        # 初始化模型权重
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
        
        # 如果没有提供位置ID，创建默认位置ID
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=inputs.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # 生成旋转位置编码
        freqs_cis = self.rotary_emb.freqs_buffer[:seq_len]
        
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
    batch_size = 2
    seq_len = 16
    feature_dim = 50  # 股票数据的特征维度
    hidden_size = 256
    num_layers = 4
    num_heads = 4
    
    # 创建测试输入
    inputs = torch.randn(batch_size, seq_len, feature_dim)
    
    # 测试Transformer模型
    print("测试股票Transformer模型...")
    model = StockTransformerModel(
        vocab_size=feature_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=seq_len,
        prediction_type="regression"
    )
    
    outputs = model(inputs)
    
    print(f"输入形状: {inputs.shape}")
    print(f"预测形状: {outputs['prediction'].shape}")
    print(f"最后隐藏状态形状: {outputs['last_hidden_state'].shape}")
    
    # 测试预测器
    print("\n测试股票价格预测器...")
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
    
    print(f"输入形状: {inputs.shape}")
    print(f"预测形状: {predictions.shape}")
    
    print("\n所有测试通过！") 