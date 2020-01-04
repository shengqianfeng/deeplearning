# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
from pystudy.common.functions import *
from pystudy.common.gradient import numerical_gradient

"""
神经网络复盘之演进：两层神经网络的随机梯度下降
    input_size:输入层的神经元数784
    hidden_size:隐藏层的神经元数50
    output_size:输出层的神经元数10
    
演示：mini-batch的大小为100，需要每次从60000个训练数据中随机取出100个数据
"""

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        # 第一层的权重（784,50）
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)  # 权重使用符合高斯分布的随机数进行初始化
        # 第一层的偏置（1,50）
        self.params['b1'] = np.zeros(hidden_size)  # 偏置使用 0进行初始化
        # 第二层的权重（50,10）
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        # 第二层的偏置（1,10）
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y

    def loss(self, x, t):
        """
        # 计算损失函数的值
        :param x: 输入数据（图像数据）
        :param t: 监督数据（正确解标签）
        """
        y = self.predict(x)
        return cross_entropy_error(y, t)

    # 计算识别精度
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy


    def gradient(self, x, t):
        """
        计算权重参数的梯度
        :param x:输入数据
        :param t:监督数据
        :return:
        """
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

    # 计算权重参数的梯度
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)# 将要被代入求值的损失函数
        # 保存梯度的字典型变量（numerical_gradient()方法的返回值）
        grads = {}
        # 第一层权重的梯度
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        # 第一层偏置的梯度
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        # 第二层权重的梯度
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        # 第二层偏置的梯度
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads


print("-----------------------开始训练随机mini SGD--------------------------------")


import numpy as np
from pystudy.dataset.mnist import load_mnist
import matplotlib.pyplot as plt


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
train_loss_list = []
# 超参数
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1


train_loss_list = []
train_acc_list = []
test_acc_list = []

# 计算抽取次数
iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
for i in range(iters_num):
    # 获取mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # 计算梯度
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch) # 高速版!
    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
        # 记录学习过程
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        # 当迭代次数i为epoch的倍数时表示基本覆盖了一轮完整数据集，统计一次数据
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)  # 计算近似总体训练数据的精确度
            test_acc = network.accuracy(x_test, t_test)  # 计算近似总体测试数据的精确度
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 绘制图形
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()