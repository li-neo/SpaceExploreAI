import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ModelArgs, Transformer
from transformers import GPT2Tokenizer
from datasets import load_dataset


# 4. 推理生成（带缓存加速）
def generate_text(model, prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids, input_ids)
            next_token = outputs[0, -1].argmax()
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
    return tokenizer.decode(input_ids[0])


if __name__ == "__main__":
    # 1. 数据加载与预处理
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 2. 初始化模型与优化器
    model = Transformer(args=ModelArgs())
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # 3. 训练循环（含梯度累积）
    gradient_accumulation_steps = 4
    for epoch in range(gradient_accumulation_steps - 1):
        for step, batch in enumerate(dataloader):
            inputs, labels = batch
            outputs = model(inputs, inputs)  # 自回归任务
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss = loss / gradient_accumulation_steps
            loss.backward()
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
