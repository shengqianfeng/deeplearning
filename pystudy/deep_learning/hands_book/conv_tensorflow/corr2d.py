#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : corr2d.py
@Author : jeffsheng
@Date : 2020/1/3
@Desc : 卷积层中的互相关（cross-correlation）运算
卷积层需要学习的参数是：卷积核和偏置大小
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


print("----------验证二维互相关运算的结果--------------")
X = tf.constant([[0,1,2], [3,4,5], [6,7,8]])
K = tf.constant([[0,1], [2,3]])
"""
<tf.Variable 'Variable:0' shape=(2, 2) dtype=float32, numpy=
array([[19., 25.],
       [37., 43.]], dtype=float32)>
"""
print(corr2d(X, K))






