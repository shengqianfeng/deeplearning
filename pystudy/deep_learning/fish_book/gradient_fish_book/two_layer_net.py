# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
from pystudy.common.functions import *
from pystudy.common.gradient import numerical_gradient


class TwoLayerNet:

    """
    input_size:输入层的神经元数784
    hidden_size:隐藏层的神经元数50
    output_size:输出层的神经元数10
    weight_init_std
    """
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {} # 保存神经网络的参数的字典型变量（实例变量）
        # 第一层的权重（784,50）
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)  # 权重使用符合高斯分布的随机数进行初始化
        # 第一层的偏置（1,50）
        self.params['b1'] = np.zeros(hidden_size)  # 偏置使用 0进行初始化
        # 第二层的权重（50,10）
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        # 第二层的偏置（1,10）
        self.params['b2'] = np.zeros(output_size)
    # 进行预测 x为输入的图像数据
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y


    # 计算损失函数的值
    # x:输入数据（图像数据）, t:监督数据（正确解标签）
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    # 计算识别精度
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy


    # 计算权重参数的梯度
    # x:输入数据, t:监督数据
    def gradient(self, x, t):
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