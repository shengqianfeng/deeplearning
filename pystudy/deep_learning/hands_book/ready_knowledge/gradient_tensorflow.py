#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : gradient.py
@Author : jeffsheng
@Date : 2020/1/2
@Desc : 介绍如何使用tensorflow2.0提供的GradientTape来自动求梯度
"""

import tensorflow as tf
print(tf.__version__)

# 创建一个4行1列的矩阵并求梯度
x = tf.reshape(tf.Variable(range(4), dtype=tf.float32), (4,1))
"""
tf.Tensor(
[[0.]
 [1.]
 [2.]
 [3.]], shape=(4, 1), dtype=float32)
"""
print(x)


with tf.GradientTape() as t:
    t.watch(x)
    # x的转置（1,4）乘以x（4,1）
    y = 2 * tf.matmul(tf.transpose(x), x)

# 利用tensorflow来求x每个元素的偏导数，也就是x的梯度，y是关于x的原函数
dy_dx = t.gradient(y, x)
"""
tf.Tensor(
[[ 0.]
 [ 4.]
 [ 8.]
 [12.]], shape=(4, 1), dtype=float32)
"""
print(dy_dx)

print("------------训练模式和预测模式----------------")
with tf.GradientTape(persistent=True) as g:
    g.watch(x)
    y = x * x
    z = y * y
    dz_dx = g.gradient(z, x)  # 108.0 (4*x^3 at x = 3)
    dy_dx = g.gradient(y, x)  # 6.0
"""
tf.Tensor(
[[  0.]
 [  4.]
 [ 32.]
 [108.]], shape=(4, 1), dtype=float32) 

tf.Tensor(
[[0.]
 [2.]
 [4.]
 [6.]], shape=(4, 1), dtype=float32)
"""
print(dz_dx,dy_dx)

print("---------对python的控制流求梯度-------------")
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c


a = tf.random.normal((1,1),dtype=tf.float32)
with tf.GradientTape() as t:
    t.watch(a)
    c = f(a)
# 对函数f(a)求梯度并与c/a比较
# tf.Tensor([[ True]], shape=(1, 1), dtype=bool)
print(t.gradient(c,a) == c/a)



