# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     modul
   Description :
   Author :       lizhenhui
   date：          2024/3/15
-------------------------------------------------
   Change Activity:
                   2024/3/15:
-------------------------------------------------
"""
import numpy as np
# import matplotlib.pyplot as plt
# import CNN.q
# CNN.q.a()
# 定义数据
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# 定义卷积层
def conv2d(x, kernel, bias):
  # 填充
  x_padded = np.pad(x, ((1, 1), (1, 1)), mode='constant')
  # 卷积
  y = np.zeros((x.shape[0], x.shape[1], x.shape[2] - kernel.shape[0] + 1, x.shape[3] - kernel.shape[1] + 1))
  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      for k in range(x.shape[2] - kernel.shape[0] + 1):
        for l in range(x.shape[3] - kernel.shape[1] + 1):
          y[i, j, k, l] = np.sum(x_padded[i, j, k:k+kernel.shape[0], l:l+kernel.shape[1]] * kernel) + bias
  # 激活函数
  return np.maximum(y, 0)

# 定义池化层
def max_pooling2d(x, pool_size):
  # 池化
  y = np.zeros((x.shape[0], x.shape[1], x.shape[2] // pool_size[0], x.shape[3] // pool_size[1]))
  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      for k in range(x.shape[2] // pool_size[0]):
        for l in range(x.shape[3] // pool_size[1]):
          y[i, j, k, l] = np.max(x[i, j, k*pool_size[0]:(k+1)*pool_size[0], l*pool_size[1]:(l+1)*pool_size[1]])
  return y

# 定义全连接层
def dense(x, w, b):
  # 矩阵乘法
  y = np.matmul(x, w) + b
  # 激活函数
  return np.maximum(y, 0)

# 定义模型
model = {
  'conv1': {
    'kernel': np.random.randn(3, 3, 1, 32),
    'bias': np.zeros(32)
  },
  'pool1': {
    'pool_size': (2, 2)
  },
  'conv2': {
    'kernel': np.random.randn(3, 3, 32, 64),
    'bias': np.zeros(64)
  },
  'pool2': {
    'pool_size': (2, 2)
  },
  'fc1': {
    'w': np.random.randn(64 * 4 * 4, 128),
    'b': np.zeros(128)
  },
  'fc2': {
    'w': np.random.randn(128, 10),
    'b': np.zeros(10)
  }
}

# 训练模型
for epoch in range(10):
  for i in range(x_train.shape[0]):
    # 前向传播
    x = x_train[i]
    x = conv2d(x, model['conv1']['kernel'], model['conv1']['bias'])
    x = max_pooling2d(x, model['pool1']['pool_size'])
    x = conv2d(x, model['conv2']['kernel'], model['conv2']['bias'])
    x = max_pooling2d(x, model['pool2']['pool_size'])
    x = np.reshape(x, (1, -1))
    x = dense(x, model['fc1']['w'], model['fc1']['b'])
    x


import numpy as np
import matplotlib.pyplot as plt

# 定义数据
iris = datasets.load_iris()

# 定义核函数
def kernel(x, y):
  return np.dot(x, y)

# 定义 SVM 模型
class SVM:
  def __init__(self, C, kernel):
    self.C = C
    self.kernel = kernel

  def fit(self, X, y):
    self.X = X
    self.y = y
    self.alpha = np.zeros(X.shape[0])

  def predict(self, X):
    y_pred = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
      for j in range(self.X.shape[0]):
        y_pred[i] += self.alpha[j] * self.y[j] * kernel(X[i], self.X[j])
    return np.sign(y_pred)

# 训练模型
model = SVM(C=1.0, kernel=kernel)
model.fit(iris.data, iris.target)

# 预测结果
predictions = model.predict(iris.data)

# 评估模型
print(classification_report(iris.target, predictions))