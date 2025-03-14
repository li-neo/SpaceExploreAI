import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 数据准备
X = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=tf.float32)
y = tf.constant([[0], [1], [1], [0]], dtype=tf.float32)

# 定义模型结构
model = Sequential([
    Dense(2, input_dim=2, activation='sigmoid'),  # 隐藏层（2个神经元）
    Dense(1, activation='sigmoid')               # 输出层
])

# 编译模型
model.compile(optimizer='sgd', loss='mean_squared_error')

# 训练模型
model.fit(X, y, epochs=10000, verbose=0)

# 测试模型
predictions = model.predict(X)
print("预测结果:", predictions.round().numpy())