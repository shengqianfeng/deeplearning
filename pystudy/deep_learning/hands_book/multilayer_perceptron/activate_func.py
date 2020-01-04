#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : activate_func.py
@Author : jeffsheng
@Date : 2020/1/2
@Desc : 使用tensorflow绘制多层感知机的激活函数
"""
# ReLU(x)=max(x,0)
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import random

def use_svg_display():
    # 用矢量图显示
    plt.savefig(fname="name", format="svg")


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


def xyplot(x_vals, y_vals, name):
    set_figsize(figsize=(5, 2.5))
    plt.plot(x_vals.numpy(), y_vals.numpy())
    plt.xlabel('x')
    plt.ylabel(name + '(x)')
    plt.show()


# 通过tf.nn提供的relu函数来绘制ReLU函数
x = tf.Variable(tf.range(-8,8, 0.1),dtype=tf.float32)
y = tf.nn.relu(x)
xyplot(x, y, 'relu')

# 绘制relu函数的导数，负数为0，正数为1
with tf.GradientTape() as t:
    t.watch(x)
    y= tf.nn.relu(x)
dy_dx = t.gradient(y, x)
xyplot(x, dy_dx, 'grad of relu')


"""
sigmoid(x)= 1/1+exp(−x)
sigmoid函数在早期的神经网络中较为普遍，但它目前逐渐被更简单的ReLU函数取代,当输入接近0时，sigmoid函数接近线性变换
"""
y = tf.nn.sigmoid(x)
xyplot(x, y, 'sigmoid')

# 绘制sigmoid函数的导数
with tf.GradientTape() as t:
    t.watch(x)
    y=y = tf.nn.sigmoid(x)
dy_dx = t.gradient(y, x)
xyplot(x, dy_dx, 'grad of sigmoid')


"""
tanh(x)= 1+exp(−2x)/1−exp(−2x)
​tanh（双曲正切）函数可以将元素的值变换到-1和1之间,
当输入接近0时，tanh函数接近线性变换。
虽然该函数的形状和sigmoid函数的形状很像，但tanh函数在坐标系的原点上对称
"""
y = tf.nn.tanh(x)
xyplot(x, y, 'tanh')

# tanh的导数绘制
with tf.GradientTape() as t:
    t.watch(x)
    y=y = tf.nn.tanh(x)
dy_dx = t.gradient(y, x)
xyplot(x, dy_dx, 'grad of tanh')






