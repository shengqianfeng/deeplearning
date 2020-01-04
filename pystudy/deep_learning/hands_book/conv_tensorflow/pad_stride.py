#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : pad_stride.py
@Author : jeffsheng
@Date : 2020/1/3
@Desc : 填充和步幅
1 填充（padding）是指在输入高和宽的两侧填充元素（通常是0元素）
2 步幅（stride）:卷积窗口从输入数组的最左上方开始，按从左往右、从上往下的顺序，依次在输入数组上滑动。我们将每次滑动的行数和列数称为步幅（stride）
"""
import  tensorflow as tf


def comp_conv2d(conv2d, X):
    X = tf.reshape(X, (1,) + X.shape + (1,))
    Y = conv2d(X)
    #input_shape = (samples, rows, cols, channels)
    return tf.reshape(Y, Y.shape[1:3])


print("--------------填充------------------------")
# 构造卷积层
# kernel_size：整数,过滤器的大小,如果为一个整数则宽和高相同
# padding： valid:表示不够卷积核大小的块,则丢弃;same表示不够卷积核大小的块就补0,所以输出和输入形状相同
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same')
X = tf.random.uniform(shape=(8,8))
# (8, 8)
print(comp_conv2d(conv2d, X).shape)


print("-------------------步幅-----------------------")
# 令高和宽上的步幅均为2，从而使输出的高和宽减半
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', strides=2)
# (4, 4)
print(comp_conv2d(conv2d, X).shape)



print("--------复杂一点-------------")
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(3,5), padding='valid', strides=(3,4))
# (2, 1)
print(comp_conv2d(conv2d, X).shape)



