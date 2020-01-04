#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : input_output_multi_channel.py
@Author : jeffsheng
@Date : 2020/1/3
@Desc : 介绍含多个输入通道或多个输出通道的卷积核
"""
import tensorflow as tf
import numpy as np
print(tf.__version__)


# 单通道二维数组卷积互相关运算
def corr2d(X, K):
    # 卷积核的高度和宽度
    h, w = K.shape
    if len(X.shape) <= 1:
        X = tf.reshape(X, (X.shape[0],1))
    # 初始化互相关运算输出矩阵
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w +1)))
    # 依次迭代每一行和列进行互相关计算
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j].assign(tf.cast(tf.reduce_sum(X[i:i+h, j:j+w] * K), dtype=tf.float32))
    return Y


"""
多输入通道互相关运算的原理：
含多个通道的输入数据与多输入通道的卷积核做二维互相关运算的输出是怎么计算的呢？
比如输入数据通道为ci，那么
1 当输入数据含多个通道时，我们需要构造一个输入通道数与输入数据的通道数相同的卷积核，从而能够与含多通道的输入数据做互相关运算
2 我们可以在各个通道上对输入的二维数组和卷积核的二维核数组做互相关运算，再将这ci个互相关运算的二维输出按通道相加，得到一个二维数组
"""

# 对每个通道做互相关运算，然后进行累加
def corr2d_multi_in(X, K):
    return tf.reduce_sum([corr2d(X[i], K[i]) for i in range(X.shape[0])],axis=0)



X = tf.constant([[[0,1,2],[3,4,5],[6,7,8]],
                 [[1,2,3],[4,5,6],[7,8,9]]])

# (2,2,2)
K = tf.constant([[[0,1],[2,3]],
                 [[1,2],[3,4]]])

print("------------多通道互相换计算--------------------")
"""
tf.Tensor(
[[ 56.  72.]
 [104. 120.]], shape=(2, 2), dtype=float32)
"""
print(corr2d_multi_in(X, K))



"""
多输出通道互相关卷积运算：
当输入通道有多个时，因为我们对各个通道的结果做了累加，所以不论输入通道数是多少，输出通道数总是为1。
那么问题来了？输出通道能不能为多通道呢，可以，卷积核除了对应输入数据的通道数ci之外，再多来几个卷积核比如个数co。
那么输入数据是(ci,h1,w2),卷积核就是(co,ci,h2,w2)
在做互相关运算时，每个输出通道上的结果由卷积核在该输出通道上的核数组与整个输入数组计算而来
"""


def corr2d_multi_in_out(X, K):
    return tf.stack([corr2d_multi_in(X, k) for k in K],axis=0)

# K(2,2,2)
# 将核数组K同K+1（K中每个元素加一）和K+2连结在一起来构造一个输出通道数为3的卷积核
K = tf.stack([K, K+1, K+2], axis=0)
# (3, 2, 2, 2)
print(K.shape)

print("---------------多通道输出-------------------")
"""
tf.Tensor(
[[[ 56.  72.]
  [104. 120.]]

 [[ 76. 100.]
  [148. 172.]]

 [[ 96. 128.]
  [192. 224.]]], shape=(3, 2, 2), dtype=float32)
"""
# 每个输入与k的每个卷积核进行互相关运算
print(corr2d_multi_in_out(X, K))


# 使用全连接层中的矩阵乘法来实现1×1 卷积

def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = tf.reshape(X,(c_i, h * w))
    K = tf.reshape(K,(c_o, c_i))
    Y = tf.matmul(K, X)
    return tf.reshape(Y, (c_o, h, w))


X = tf.random.uniform((3,3,3))
K = tf.random.uniform((2,3,1,1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)

# tf.Tensor(True, shape=(), dtype=bool)
print(tf.norm(Y1-Y2) < 1e-6)
