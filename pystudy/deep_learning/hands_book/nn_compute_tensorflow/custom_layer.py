#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : custom_layer.py
@Author : jeffsheng
@Date : 2020/1/3
@Desc : 虽然tf.keras提供了大量常用的层，但有时候我们依然希望自定义层
"""

import tensorflow as tf
import numpy as np
print(tf.__version__)


X = tf.random.uniform((2,20))
print("--------定义一个不含模型参数的自定义层----------")


class CenteredLayer(tf.keras.layers.Layer):
    """
    CenteredLayer类通过继承tf.keras.layers.Layer类自定义了一个将输入减掉均值后输出的层
    """
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return inputs - tf.reduce_mean(inputs)


# 1实例化这个层，然后做前向计算
layer = CenteredLayer()
# tf.Tensor([-2 -1  0  1  2], shape=(5,), dtype=int32)
print(layer(np.array([1,2,3,4,5])))

# 2构造更加复杂的模型
net = tf.keras.models.Sequential()
net.add(tf.keras.layers.Flatten())
net.add(tf.keras.layers.Dense(20))
net.add(CenteredLayer())

Y = net(X)
"""
tf.Tensor(
[[-0.41374725  0.52120537  0.24625513 -0.1075322  -0.33142442  0.06008503
   0.3076693   0.4108904  -1.8816154  -0.03293392  1.71756     0.0799534
  -0.99712205 -0.53492     0.21637893  0.8344747  -0.06739563 -0.34540647
   0.70946807  0.35109732]
 [-0.7740419  -0.31630507  0.05992274 -0.33549574  0.00545774 -0.12231851
   0.32372993  1.12497    -0.8291088   0.483728    1.0563822  -0.1725609
  -0.06369461 -0.5267022   0.18054181 -0.33663005 -0.84464395 -0.63882303
   0.6223497   0.3603031 ]], shape=(2, 20), dtype=float32)
"""
print(Y)
# 打印自定义层各个输出的均值。因为均值是浮点数，所以它的值是一个很接近0的数
# tf.Tensor(2.682209e-08, shape=(), dtype=float32)
print(tf.reduce_mean(Y))


print("--------定义含模型参数的自定义层-----------")


class myDense(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):     # 这里 input_shape 是第一次运行call()时参数inputs的形状
        self.w = self.add_weight(name='w', shape=[input_shape[-1], self.units], initializer=tf.random_normal_initializer())
        self.b = self.add_weight(name='b', shape=[self.units], initializer=tf.zeros_initializer())

    def call(self, inputs):
        y_pred = tf.matmul(inputs, self.w) + self.b
        return y_pred


# 接使用自定义层做前向计算
dense = myDense(3)
# 先调用build再调call
print(dense(X))
# 模型参数可以通过训练学出
print(dense.get_weights())


# 也可以使用自定义层构造模型
net = tf.keras.models.Sequential()
net.add(myDense(8))
net.add(myDense(1))

print(net(X))





