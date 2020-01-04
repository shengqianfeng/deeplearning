#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2020/1/1 0001 下午 9:23 
# @Author : jeffsmile 
# @File : linear-regression-scratch.py
# @desc :线性回归从零开始实现

import tensorflow as tf
print(tf.__version__)
from matplotlib import pyplot as plt
import random

# 特征维度
num_inputs = 2
# 样本大小
num_examples = 1000
# 真实权重
true_w = [2, -3.4]
# 真实偏置
true_b = 4.2
# 随机生成一个（1000,2）的样本特征
features = tf.random.normal((num_examples, num_inputs), stddev = 1)
# 由真实权重偏置生成对应features的（1000,1）的标签
labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b
# 给每一个标签都加入噪声，是一个标准差为0.01的随机值
labels += tf.random.normal(labels.shape, stddev=0.01)

"""
打印特征值和标签第一个元素
tf.Tensor([-1.1249537   0.65576553], shape=(2,), dtype=float32) 
tf.Tensor(-0.28441852, shape=(), dtype=float32)
"""
print(features[0], labels[0])

# 设置画布大小
def set_figsize(figsize=(3.5, 2.5)):
    plt.rcParams['figure.figsize'] = figsize


set_figsize()
# 观察第二个特征features[:, 1]和标签 labels 两者之间的线性关系
plt.scatter(features[:, 1], labels, 1)
plt.show()

print("-----------读取数据----------------")
import numpy as np

# 定义一个函数：它每次返回batch_size（批量大小）个随机样本的特征和标签
def data_iter(batch_size, features, labels):
    features = np.array(features)
    labels = np.array(labels)
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = np.array(indices[i: min(i + batch_size, num_examples)])
        yield features[j], labels[j]

# 读取第一个小批量数据样本特征、标签并打印
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break

print("------------初始化模型参数--------------")
# 将权重初始化成均值为0、标准差为0.01的正态随机数，偏差则初始化成0
w = tf.Variable(tf.random.normal((num_inputs, 1), stddev=0.01))
b = tf.Variable(tf.zeros((1,)))


# 线性回归的矢量计算矩阵乘法
def linreg(X, w, b):
    return tf.matmul(X, w) + b


print("---------定义损失函数------------------")
# 定义平方损失函数作为线性回归的损失函数
def squared_loss(y_hat, y):
    """
    :param y_hat: 预测值(10,1)
    :param y: 真实值 形状(10,)
    :return:
    """
    return (y_hat - tf.reshape(y, y_hat.shape)) ** 2 /2


print("-----------定义优化算法------------")
# 小批量随机梯度下降算法，它通过不断迭代模型参数来优化损失函数
def sgd(params, lr, batch_size):
    """
    将小批量的梯度和除以batchsize得到的平均值作为本次计算出梯度值，然后相减梯度下降
    :param params:权重和偏置 list结构
    :param lr:
    :param batch_size:
    :return:
    l:当前小批量样本集计算得出的损失值(10,1)

    ☆数学公式指导理解：
    t.gradient(target,source)返回一个与source参数一一对应的梯度值列表，代表对应source变量的梯度
    是batchsize这批样本的梯度参数矩阵之和
    """
    for param in params:
#         param[:] = param - lr * t.gradient(l, param) / batch_size
        param.assign_sub(lr * t.gradient(l, param) / batch_size)


print("--------训练模型-------------------")
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        """
        GradientTape：梯度带
        参数：
            persistent: 布尔值，用来指定新创建的gradient tape是否是可持续性的。
                        默认是False，意味着只能够调用一次gradient（）函数,
                        设定persistent为True，便可以在这个上面反复调用gradient（）函数
             watch(tensor)：确保某个tensor被tape追踪，其参数tensor: 一个Tensor或者一个Tensor列表           
        """
        with tf.GradientTape(persistent=True) as t:
            t.watch([w,b])
            # 每个小批量batchsize=10的损失值，形状（10,1）
            l = loss(net(X, w, b), y)
        sgd([w, b], lr, batch_size)
    # train_l（1000,1）
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, tf.reduce_mean(train_l)))


# 观察学习到的参数和真实参数之间的差距
"""
结果接近：
[2, -3.4] 
<tf.Variable 'Variable:0' shape=(2, 1) dtype=float32, numpy=array([[ 1.9994726],[-3.3999593]], dtype=float32)>
"""
print(true_w, '\n', w)
"""
4.2 
<tf.Variable 'Variable:0' shape=(1,) dtype=float32, numpy=array([4.1996937], dtype=float32)>
"""
print(true_b, '\n', b)
