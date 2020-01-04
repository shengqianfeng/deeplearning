#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : model_param_get_init_share.py.py
@Author : jeffsheng
@Date : 2020/1/3
@Desc : 模型参数的访问、初始化和共享
"""
import tensorflow as tf
import numpy as np
print(tf.__version__)


net = tf.keras.models.Sequential()
net.add(tf.keras.layers.Flatten())
net.add(tf.keras.layers.Dense(256,activation=tf.nn.relu))
net.add(tf.keras.layers.Dense(10))

X = tf.random.uniform((2,20))
Y = net(X)
print(Y)

print("-----------访问模型参数------------------")
"""
对于使用Sequential类构造的神经网络，我们可以通过weights属性来访问网络任一层的权重.
对于Sequential实例中含模型参数的层，我们可以通过tf.keras.Model类的weights属性来访问该层包含的所有参数
"""
print(net.weights[0], type(net.weights[0]))

print("------------初始化模型参数---------------")
"""
将权重参数初始化成均值为0、标准差为0.01的正态分布随机数，并依然将偏差参数清零
"""


class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(
            units=10,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )
        self.d2 = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.ones_initializer(),
            bias_initializer=tf.ones_initializer()
        )

    def call(self, input):
        output = self.d1(input)
        output = self.d2(output)
        return output


net = Linear()
print(net(X))
print(net.get_weights())


print("--------自定义初始化参数-----------------")
"""
可以使用tf.keras.initializers类中的方法实现自定义初始化
"""


def my_init():
    return tf.keras.initializers.Ones()


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, kernel_initializer=my_init()))

Y = model(X)
print(Y)
print(model.weights[0])





