import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        
        # 初始化权重矩阵，随机小数

        #
        #     input:     x1     x2
        #     hidden:    h1     h2
        #     output:        y
        #

        self.hidden_output = None
        self.hidden_input = None
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.bias1 = np.random.randn(hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias2 = np.random.randn(output_size)
     #  激活函数 
     # 激活函数	 输出范围	导数特性	适用场景
     #  Sigmoid	(0, 1)	导数易消失	二分类输出层
     #  Tanh	(-1, 1)	梯度比 Sigmoid 更强	隐藏层
     #  ReLU	[0, +∞)	简单且缓解梯度消失	隐藏层（最常用）
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # sigmod 激活函数的导数

    def sigmoid_derivative(self, s):
        return s * (1 - s)
    
    # 前向网络
    def  forward(self, x):
        # 第一层： 输入层 -> 隐藏层
        self.hidden_input = np.dot(x, self.weights1) + self.bias1
        self.hidden_output = self.sigmoid(self.hidden_input)

        # 第二层： 隐藏层  -> 输出层
        self.output_input = np.dot(self.hidden_output, self.weights2) +  self.bias2
        self.output = self.sigmoid(self.output_input)

        return self.output
    
    # 训练
    # x: 输入参数
    # y: 输出
    # epochs: 训练次数
    # lr: 学习因子
    def train(self, x, y, epochs = 10000, lr = 0.1):
        for epoch in range(epochs):
            # 前向网络传播计算

            output = self.forward(x)

            # 反向网络传播计算， 梯度下降
            output_error = y - output
            # 输出层 -> 隐藏层
            # 输出层delta
            output_delta = output_error * self.sigmoid_derivative(output)

            #计算隐藏层error, delta
            hidden_error = output_delta.dot(self.weights2.T)
            hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

            # 更新权重和偏置

            self.weights2 += self.hidden_output.T.dot(output_delta) * lr

            # ​数学原理：
            # 偏置的梯度是输出层误差在各样本上的和（因偏置对每个样本的误差均有贡献）：
            # np.sum(output_delta, axis=0)：沿样本维度（axis=0）求和，得到形状 [1, n_output]。
            # keepdims=True：保持维度，确保与 bias2 的维度一致。
            # * lr：乘以学习率。
            self.bias2 += np.sum(output_delta, axis=0) * lr
            # self.bias2 += np.sum(output_delta, axis=0, keepdims=True) * lr

            #更新输入层和隐藏层的权重和偏置

            self.weights1 += x.T.dot(hidden_delta) * lr
            self.bias1 += np.sum(hidden_delta, axis=0) * lr
             

             # 每一百轮打印一次损失

            # if epoch % 200 == 0:
                #                 计算均方误差损失**
                # loss = np.mean(np.square(y - output))
                # ​步骤拆解:
                # y - output：计算每个样本的预测值与真实值的差值（形状与 y 和 output 相同）。
                # np.square(...)：对差值逐元素平方，放大较大误差（如误差为2 → 平方后为4）。
                # np.mean(...)：对所有样本的平方误差取平均，得到标量损失值

                # loss = np.mean(np.square(y - output))
                # print(f"Epoch:{epoch}, Loss:{loss:.4f}")

#   取异或
x_input = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y_output = np.array([[0], [1], [1], [0]])

nn = SimpleNeuralNetwork(input_size=2, hidden_size=2, output_size=1)
nn.train(x_input, y_output, epochs=10000, lr=0.15)

predictions = nn.forward(x_input)

print(f"\nPredictions:\n{predictions}")




