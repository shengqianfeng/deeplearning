#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : linear_regression_keras.py
@Author : jeffsheng
@Date : 2020/1/2
@Desc : 介绍如何使用tensorflow2.0推荐的keras接口更方便地实现线性回归的训练
线性回归是单层神经网络，输出层是一个全连接层
"""

import tensorflow as tf
print("---------生成数据集-----------------")

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = tf.random.normal(shape=(num_examples, num_inputs), stddev=1)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += tf.random.normal(labels.shape, stddev=0.01)

from tensorflow import data as tfdata
print("----------读取数据------------------------")
batch_size = 10
# 将训练数据的特征和标签组合
dataset = tfdata.Dataset.from_tensor_slices((features, labels))
# 随机读取小批量 shuffle 的 buffer_size 参数应大于等于样本数
dataset = dataset.shuffle(buffer_size=num_examples)
# batch函数可以指定batch_size的分割大小
dataset = dataset.batch(batch_size)
data_iter = iter(dataset)

# 打印第一个数据集及标签
for X, y in data_iter:
    print(X, y)
    break

# 官方推荐写法
for (batch, (X, y)) in enumerate(dataset):
    print(X, y)
    break


print("-------------定义模型和初始化参数------------------------")
from tensorflow import keras
# Tensorflow 2.0推荐使用Keras定义网络
from tensorflow.keras import layers
from tensorflow import initializers as init
# Sequential实例可以看作是一个串联各个层的容器
model = keras.Sequential()
# kernel_initializer:设置权重的初始化方式
# bias_initializer:设置偏置的初始化方式，默认会初始化为0
# RandomNormal(stddev=0.01):指定权重参数每个元素将在初始化时随机采样于均值为0、标准差为0.01的正态分布
# 在构造模型时，我们在该容器中依次添加层。线性回归，输入层与输出层等效为一层全连接层keras.layers.Dense()
model.add(layers.Dense(1, kernel_initializer=init.RandomNormal(stddev=0.01)))


print("---------------定义损失函数------------------")
"""
Tensoflow在losses模块中提供了各种损失函数和自定义损失函数的基类，并直接使用它的均方误差损失作为模型的损失函数
"""
from tensorflow import losses
loss = losses.MeanSquaredError()

print("-----------定义优化算法----------------")
"""
我们无需自己实现小批量梯度下降算法，tensorflow.keras.optimizers 模块提供了很多常用的优化算法比如SGD、Adam和RMSProp等
"""
from tensorflow.keras import optimizers
trainer = optimizers.SGD(learning_rate=0.03)


print("------------训练模型--------------------")
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for (batch, (X, y)) in enumerate(dataset):
        with tf.GradientTape() as tape:
            l = loss(model(X, training=True), y)
        # 执行tape.gradient获得动态图中各变量梯度 ，通过model.trainable_variables 找到需要更新的变量
        grads = tape.gradient(l, model.trainable_variables)
        # trainer.apply_gradients 更新权重，完成一步训练
        trainer.apply_gradients(zip(grads, model.trainable_variables))
    # 每次epoach后在整个训练集上进行验证，求得平均的损失值
    l = loss(model(features), labels)
    print('epoch %d, loss: %f' % (epoch, l.numpy().mean()))


print("--------比较学到的模型参数和真实的模型参数----")
"""
权重：
[2, -3.4]-----[[ 2.0001652] [-3.4007895]]
偏置：
4.2-----[4.198627]
"""
print(true_w, model.get_weights()[0])
print(true_b, model.get_weights()[1])








