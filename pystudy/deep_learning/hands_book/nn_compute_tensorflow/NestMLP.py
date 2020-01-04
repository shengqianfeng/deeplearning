#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : NestMLP.py
@Author : jeffsheng
@Date : 2020/1/3
@Desc : 因为FancyMLP和Sequential类都是tf.keras.Model类的子类，所以我们可以嵌套调用它们
"""
import tensorflow as tf
print(tf.__version__)

class FancyMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.rand_weight = tf.constant(tf.random.uniform((20,20)))
        self.dense = tf.keras.layers.Dense(units=20, activation=tf.nn.relu)
    # tf.norm用来计算范数，缺省值是 'euclidean'，如果张量是一个矩阵，则相当于 Frobenius 范数
    def call(self, inputs):
        x = self.flatten(inputs)
        x = tf.nn.relu(tf.matmul(x, self.rand_weight) + 1)
        x = self.dense(x)
        while tf.norm(x) > 1:
            x /= 2
        if tf.norm(x) < 0.8:
            x *= 10
        return tf.reduce_sum(x)


class NestMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.net = tf.keras.Sequential()
        self.net.add(tf.keras.layers.Flatten())
        self.net.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        self.net.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
        self.dense = tf.keras.layers.Dense(units=16, activation=tf.nn.relu)


    def call(self, inputs):
        return self.dense(self.net(inputs))

net = tf.keras.Sequential()
net.add(NestMLP())
net.add(tf.keras.layers.Dense(20))
net.add(FancyMLP())

X = tf.random.uniform((2,20))
# tf.Tensor(20.01051, shape=(), dtype=float32)
print(net(X))





