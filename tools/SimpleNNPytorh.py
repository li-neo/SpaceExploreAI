import torch
import torch.nn as nn
import torch.optim as optim


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=output_size)

    # 前向网络
    def forward(self, p):
        # 将sigmoid激活函数改为Relu函数
        # 
        p = torch.sigmoid(self.fc1(p))

        # 使用ReLu函数错误率很高
        # p = torch.relu(self.fc1(p))
        # 因为使用BCEWithLogisLoss 直接输出logis
        p = torch.sigmoid(self.fc2(p))
        # p = self.fc2(p)
        return p

x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)


NN_model = SimpleNN(input_size=x.shape[1], hidden_size=x.shape[1], output_size=y.shape[1])
# 二分类的损失函数
criterion = nn.BCELoss()
#使用梯度下降算法优化模型参数， model.parameters() 获取模型中所有可学习的参数（权重和偏置）
# lr： 学习率，控制参数更新的步长，学习率过大导致震荡和发散； 过小收敛很慢，容易陷入局部最优
# 扩展优化器： 带动量的SGD， 加速收敛并减少震荡
#            自适应优化器： 如Adam：结合动量和自适应学习率，适合复杂非凸优化
optimizer = optim.SGD(NN_model.parameters(), lr=0.01, momentum=0.9)
num_epochs = 10000
for epoch in range(num_epochs):
    # 前向传播
    output = NN_model(x)
    loss = criterion(output, y)

    #反向传播与优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward()       #
    optimizer.step()
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")


with torch.no_grad():
    test_output = NN_model(x)
    predicted = (test_output > 0.5).float()
    accuracy = (predicted == y).sum().item() / y.numel()
    print(f"训练完成，准确率：{accuracy:.2f}%")

