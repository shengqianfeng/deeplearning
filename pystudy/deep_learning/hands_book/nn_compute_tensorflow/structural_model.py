#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : structural_model.py
@Author : jeffsheng
@Date : 2020/1/3
@Desc : 基于tf.keras.model构造模型
"""

import tensorflow as tf
import numpy as np
print(tf.__version__)

"""
MLP类中无须定义反向传播函数。
系统将通过自动求梯度而自动生成反向传播所需的backward函数
"""
class MLP(tf.keras.Model):
    """
    继承tf.keras.Model类构造多层感知机
    MLP类重载了tf.keras.Model类的__init__函数和call函数
    """
    # 创建模型参数
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()    # Flatten层将除第一维（batch_size）以外的维度展平
        self.dense1 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    # 定义正向传播
    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        output = self.dense2(x)
        return output


X = tf.random.uniform((2,20))
net = MLP()
# net(X)将调用MLP类定义的call函数来完成前向计算
"""
tf.Tensor(
[[-0.24679185  0.17069653 -0.09609146 -0.08310401 -0.2591616  -0.05466332
   0.10410955  0.12337537 -0.29189035  0.03359837]
 [-0.18600827  0.2402858  -0.2907969  -0.00555098 -0.557043   -0.2994769
   0.31195667  0.11417881 -0.54367656 -0.11149988]], shape=(2, 10), dtype=float32)
"""
print(net(X))

