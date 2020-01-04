# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
import numpy as np
from pystudy.common.functions import softmax, cross_entropy_error
from pystudy.common.gradient import numerical_gradient

"""
神经网络复盘之演进：简单的单层神经网络求梯度
输入层：2个神经元
输出层：2个神经元

"""

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

# 输入参数
x = np.array([0.6, 0.9])
# 输入参数正确结果标签
t = np.array([0, 0, 1])

net = simpleNet()
"""
与下边的lamada等价
def f(W):
... return net.loss(x, t)
"""
f = lambda w: net.loss(x, t)
"""
进行一次梯度求解
[[ 0.13513108  0.12719361 -0.26232469]
 [ 0.20269662  0.19079042 -0.39348704]]
"""
dW = numerical_gradient(f, net.W)
print(dW)
