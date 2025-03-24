"""
辅助函数和工具类
用于模型训练、评估和推理
"""

import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image


def set_seed(seed):
    """
    设置随机种子以确保结果可复现
    
    参数:
        seed (int): 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_checkpoint(model, checkpoint_path, optimizer=None, device=None):
    """
    加载模型检查点
    
    参数:
        model: 模型实例
        checkpoint_path (str): 检查点文件路径
        optimizer: 优化器实例，可选
        device: 设备，可选
        
    返回:
        int: 当前epoch
    """
    if not os.path.exists(checkpoint_path):
        print(f"检查点 {checkpoint_path} 不存在")
        return 0
    
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    print(f"从检查点 {checkpoint_path} 加载的模型 (epoch {epoch})")
    
    return epoch


def save_checkpoint(model, optimizer, epoch, save_path, metrics=None):
    """
    保存模型检查点
    
    参数:
        model: 模型实例
        optimizer: 优化器实例
        epoch (int): 当前epoch
        save_path (str): 保存路径
        metrics (dict): 模型指标，可选
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, save_path)
    print(f"模型保存到 {save_path}")


def visualize_attention(image, attention_weights, save_path=None):
    """
    可视化注意力权重
    
    参数:
        image: 输入图像
        attention_weights: 注意力权重
        save_path (str): 保存路径，可选
    """
    plt.figure(figsize=(10, 5))
    
    # 显示原始图像
    plt.subplot(1, 2, 1)
    if isinstance(image, torch.Tensor):
        # 转换PyTorch张量为NumPy数组
        if image.dim() == 4:  # [batch, channels, height, width]
            image = image[0]  # 取第一张图片
        
        image = image.permute(1, 2, 0).cpu().numpy()  # [height, width, channels]
        
        # 反标准化（假设使用了ImageNet标准化）
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        
        # 确保值在 [0, 1] 范围内
        image = np.clip(image, 0, 1)
    
    plt.imshow(image)
    plt.title('原始图像')
    plt.axis('off')
    
    # 显示注意力图
    plt.subplot(1, 2, 2)
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()
    
    # 如果注意力权重是多维的，取平均值
    if attention_weights.ndim > 2:
        attention_weights = np.mean(attention_weights, axis=0)
    
    plt.imshow(attention_weights, cmap='viridis')
    plt.title('注意力权重')
    plt.colorbar()
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"注意力可视化已保存到 {save_path}")
    
    plt.show()


def tokenize_prompt(prompt, tokenizer, max_length=100):
    """
    对提示词进行标记化处理
    
    参数:
        prompt (str): 输入提示词
        tokenizer: 分词器
        max_length (int): 最大长度
        
    返回:
        tensor: 标记化后的提示词
    """
    if hasattr(tokenizer, 'encode_plus'):
        # 使用Transformers库的tokenizer
        encoded = tokenizer.encode_plus(
            prompt,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoded['input_ids']
    else:
        # 使用自定义tokenizer
        tokens = tokenizer.tokenize(prompt)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # 添加特殊标记（如果需要）
        if hasattr(tokenizer, 'cls_token_id') and hasattr(tokenizer, 'sep_token_id'):
            token_ids = [tokenizer.cls_token_id] + token_ids + [tokenizer.sep_token_id]
        
        # 填充或截断
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids += [0] * (max_length - len(token_ids))
        
        return torch.tensor([token_ids], dtype=torch.long)


def decode_prompt(token_ids, tokenizer):
    """
    将token id解码为提示词
    
    参数:
        token_ids (tensor): token id序列
        tokenizer: 分词器
        
    返回:
        str: 解码后的提示词
    """
    if hasattr(tokenizer, 'decode'):
        # 使用Transformers库的tokenizer
        return tokenizer.decode(token_ids.squeeze().tolist(), skip_special_tokens=True)
    else:
        # 使用自定义tokenizer
        tokens = [tokenizer.convert_ids_to_tokens(idx) for idx in token_ids.squeeze().tolist()]
        
        # 过滤特殊标记
        special_tokens = {'<pad>', '<sos>', '<eos>', '<unk>'}
        tokens = [token for token in tokens if token not in special_tokens]
        
        # 合并tokens
        return ' '.join(tokens)


def prepare_image(image_path, transform=None):
    """
    准备图像用于模型输入
    
    参数:
        image_path (str): 图像路径
        transform (callable): 图像转换函数，可选
        
    返回:
        tensor: 处理后的图像张量
    """
    try:
        image = Image.open(image_path).convert('RGB')
        
        if transform is None:
            # 默认转换
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        image_tensor = transform(image).unsqueeze(0)  # 添加批次维度
        return image_tensor
    
    except Exception as e:
        print(f"处理图像时出错: {e}")
        return None 