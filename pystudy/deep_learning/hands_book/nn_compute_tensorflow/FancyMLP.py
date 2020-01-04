#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : FancyMLP.py
@Author : jeffsheng
@Date : 2020/1/3
@Desc : 通过继承tf.keras.model灵活构造复杂模型
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


if __name__ == '__main__':
    X = tf.random.uniform((2,20))
    # 测试该模型的随机初始化和前向计算
    net = FancyMLP()
    # tf.Tensor(20.200562, shape=(), dtype=float32)
    print(net(X))


