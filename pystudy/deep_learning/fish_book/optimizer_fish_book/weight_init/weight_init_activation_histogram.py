# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

"""
# author jeffsheng
# data 20191222.0935
# title sigmoid函数的权重初始值
    观察权重初始值是如何影响隐藏层的激活值(激活函数的输出数据)的分布的
"""
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)
    
input_data = np.random.randn(1000, 100)  # 1000个数据 （1000,100）
node_num = 100  # 各隐藏层的节点（神经元）数
hidden_layer_size = 5  # 隐藏层有5层
activations = {}  # 激活值的结果保存在这里

x = input_data
# 计算5个隐藏层输出的激活值，每一层输出激活值为(1000,100)
for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1] # 将上一层的激活值作为下一层的输入

    # 改变初始值进行实验！
    w = np.random.randn(node_num, node_num) * 1 # 初始化当前隐藏层的权重（100,100）
    # w = np.random.randn(node_num, node_num) * 0.01
    # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
    # w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)


    a = np.dot(x, w) # 神经元计算结果为（1000,100）


    # 将激活函数的种类也改变，来进行实验！
    z = sigmoid(a) # 激活（1000,100）神经元结果矩阵，输出（1000,100）的激活值矩阵
    # z = ReLU(a)
    # z = tanh(a)

    activations[i] = z  # 保存每一层激活函数的结果值

# 将保存在activations中的各层数据画成直方图
# 观察使用标准差为1的高斯分布作为权重初始值时的各层激活值的分布
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([], [])
    # plt.xlim(0.1, 1)
    # plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()
