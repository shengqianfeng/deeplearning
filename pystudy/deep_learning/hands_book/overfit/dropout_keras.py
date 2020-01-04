#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : dropout_keras.py
@Author : jeffsheng
@Date : 2020/1/2 0002
@Desc : dropout丢弃法的keras简单实现
    在Tensorflow2.0中，我们只需要在全连接层后添加Dropout层并指定丢弃概率。
在训练模型时，Dropout层将以指定的丢弃概率随机丢弃上一层的输出元素；
在测试模型时（即model.eval()后），Dropout层并不发挥作用
"""
from tensorflow.keras.datasets import fashion_mnist
import tensorflow as tf
import numpy as np
from tensorflow import keras, nn, losses
from tensorflow.keras.layers import Dropout, Flatten, Dense

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(256,activation='relu'),
    Dropout(0.2),
    keras.layers.Dense(256,activation='relu'),
    Dropout(0.5),
    keras.layers.Dense(10,activation=tf.nn.softmax)
])

batch_size=256
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = tf.cast(x_train, tf.float32) / 255 #在进行矩阵相乘时需要float型，故强制类型转换为float型
x_test = tf.cast(x_test,tf.float32) / 255 #在进行矩阵相乘时需要float型，故强制类型转换为float型
train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)


print("----------训练并测试模型---------------------------")
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train,y_train, epochs=5, batch_size=256, validation_data=(x_test, y_test), validation_freq=1)


