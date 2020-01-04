#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : pool.py
@Author : jeffsheng
@Date : 2020/1/3
@Desc : 介绍池化（pooling）层，它的提出是为了缓解卷积层对位置的过度敏感性
1 池化层直接计算池化窗口内元素的最大值或者平均值
2 该运算也分别叫做最大池化或平均池化
"""
import tensorflow as tf


def pool2d(X, pool_size, mode='max'):
    """
    :param X: 池化输入数据
    :param pool_size:  池化窗口大小
    :param mode: 池化模式：最大或平均
    :return:
    """
    p_h, p_w = pool_size
    Y = tf.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w +1))
    Y = tf.Variable(Y)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i,j].assign(tf.reduce_max(X[i:i+p_h, j:j+p_w]))
            elif mode =='avg':
                Y[i,j].assign(tf.reduce_mean(X[i:i+p_h, j:j+p_w]))
    return Y


print("--------验证二维最大池化层的输出---------")
X = tf.constant([[0,1,2],[3,4,5],[6,7,8]],dtype=tf.float32)
"""
<tf.Variable 'Variable:0' shape=(2, 2) dtype=float32, numpy=
array([[4., 5.],
       [7., 8.]], dtype=float32)>
"""
print(pool2d(X, (2,2)))

print("--------验证二维平均池化层的输出---------")
"""
<tf.Variable 'Variable:0' shape=(2, 2) dtype=float32, numpy=
array([[2., 3.],
       [5., 6.]], dtype=float32)>
"""
print(pool2d(X, (2,2), 'avg'))


print("-------------------池化层输入的填充和步幅---------------")
"""
同卷积层一样，池化层也可以在输入的高和宽两侧的填充并调整窗口的移动步幅来改变输出形状。
池化层填充和步幅与卷积层填充和步幅的工作机制一样
"""

# 构造一个形状为(1, 1, 4, 4)的输入数据，前两个维度分别是批量和通道
# tensorflow默认数据类型为'channels_last'，所以这里使用(1,4,4,1)而不是(1,1,4,4)
X = tf.reshape(tf.constant(range(16)), (1,4,4,1))
"""
tf.Tensor(
[[[[ 0]
   [ 1]
   [ 2]
   [ 3]]

  [[ 4]
   [ 5]
   [ 6]
   [ 7]]

  [[ 8]
   [ 9]
   [10]
   [11]]

  [[12]
   [13]
   [14]
   [15]]]], shape=(1, 4, 4, 1), dtype=int32)
"""
print(X)


# 通过nn模块里的二维最大池化层MaxPool2D来演示池化层填充和步幅的工作机制
"""
默认情况下，MaxPool2D实例里步幅和池化窗口形状相同
下面使用形状为(3, 3)的池化窗口，默认获得形状为(3, 3)的步幅
"""
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3,3])
# tf.Tensor([[[[10]]]], shape=(1, 1, 1, 1), dtype=int32)
print(pool2d(X))



#I guess no custom padding settings in keras.layers?
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3,3],padding='same',strides=2)
"""
tf.Tensor(
[[[[10]
   [11]]

  [[14]
   [15]]]], shape=(1, 2, 2, 1), dtype=int32)
"""
print(pool2d(X))


print("-------------多通道输入数据的池化-------------")
"""
在处理多通道输入数据时，池化层对每个输入通道分别池化，
而不是像卷积层那样将各通道的输入按通道相加。
这意味着池化层的输出通道数与输入通道数相等。
"""
X = tf.stack([X, X+1], axis=3)
X = tf.reshape(X, (2,4,4,1))
# (2, 4, 4, 1)
print(X.shape)

# pool_size为整数，则两个维度将使用同一窗口
pool2d = tf.keras.layers.MaxPool2D(3, padding='same', strides=2)
"""
tf.Tensor(
[[[[ 5]
   [ 6]]

  [[ 7]
   [ 8]]]


 [[[13]
   [14]]

  [[15]
   [16]]]], shape=(2, 2, 2, 1), dtype=int32)
"""
print(pool2d(X))

