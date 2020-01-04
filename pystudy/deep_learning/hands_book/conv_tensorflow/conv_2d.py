#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : conv_2d.py
@Author : jeffsheng
@Date : 2020/1/3
@Desc : 基于corr2d函数来实现一个自定义的二维卷积层
"""

import tensorflow as tf
def corr2d(X, K):
    """
    定义二维互相关运算函数
    :param X:输入数组
    :param K: 核数组
    :return:二维互相关的运算结果
    """
    h, w = K.shape
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w +1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j].assign(tf.cast(tf.reduce_sum(X[i:i+h, j:j+w] * K), dtype=tf.float32))
    return Y


"""
1 二维卷积层将输入和卷积核做互相关运算，并加上一个标量偏差来得到输出
2 卷积层的模型参数包括了卷积核和标量偏差。在训练模型的时候，通常我们先对卷积核随机初始化，然后不断迭代卷积核和偏差
"""
class Conv2D(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, kernel_size):
        self.w = self.add_weight(name='w',
                                shape=kernel_size,
                                initializer=tf.random_normal_initializer())
        self.b = self.add_weight(name='b',
                                shape=(1,),
                                initializer=tf.random_normal_initializer())
    def call(self, inputs):
        return corr2d(inputs, self.w + self.b)


print("---------图像边缘检测------------")
# 构造一张（6,8）图像，它中间4列为黑（0），其余为白（1）
X = tf.Variable(tf.ones((6,8)))
X[:, 2:6].assign(tf.zeros(X[:,2:6].shape))
"""
<tf.Variable 'Variable:0' shape=(6, 8) dtype=float32, numpy=
array([[1., 1., 0., 0., 0., 0., 1., 1.],
       [1., 1., 0., 0., 0., 0., 1., 1.],
       [1., 1., 0., 0., 0., 0., 1., 1.],
       [1., 1., 0., 0., 0., 0., 1., 1.],
       [1., 1., 0., 0., 0., 0., 1., 1.],
       [1., 1., 0., 0., 0., 0., 1., 1.]], dtype=float32)>
"""
print(X)

# 构造一个高和宽分别为1和2的卷积核K。
# 当它与输入做互相关运算时，如果横向相邻元素相同，输出为0；否则输出为非0
K = tf.constant([[1,-1]], dtype = tf.float32)

"""
结果显示：从白到黑的边缘和从黑到白的边缘分别检测成了1和-1。其余部分的输出全是0
<tf.Variable 'Variable:0' shape=(6, 7) dtype=float32, numpy=
array([[ 0.,  1.,  0.,  0.,  0., -1.,  0.],
       [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
       [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
       [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
       [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
       [ 0.,  1.,  0.,  0.,  0., -1.,  0.]], dtype=float32)>
"""
Y = corr2d(X, K)
print(Y)




print("--------learn kernel by data-------------------")
X = tf.reshape(X, (1, 6, 8, 1))
Y = tf.reshape(Y, (1, 6, 7, 1))
print(Y)

# 使用tf.keras.layers提供的Conv2D类可以自动求梯度
conv2d = tf.keras.layers.Conv2D(1, (1,2))
print(Y.shape)

# 预测卷积结果
Y_hat = conv2d(X)
for i in range(10):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(conv2d.weights[0])
        Y_hat = conv2d(X)
        # 均差平方作为损失函数
        l = (abs(Y_hat - Y)) ** 2
        dl = g.gradient(l, conv2d.weights[0])
        lr = 3e-2
        update = tf.multiply(lr, dl)
        updated_weights = conv2d.get_weights()
        updated_weights[0] = conv2d.weights[0] - update
        conv2d.set_weights(updated_weights)

        if (i + 1)% 2 == 0:
            print('batch %d, loss %.3f' % (i + 1, tf.reduce_sum(l)))


# 观察学习到的核数组
print(tf.reshape(conv2d.get_weights()[0],(1,2)))
