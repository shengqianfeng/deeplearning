#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : multiayer_perceptron_keras.py
@Author : jeffsheng
@Date : 2020/1/2
@Desc : 使用keras来实现多层感知机
"""

import tensorflow as tf
from tensorflow import keras
import sys
sys.path.append("..")
from tensorflow import keras
fashion_mnist = keras.datasets.fashion_mnist

print("---------定义模型---------")
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu',),
    tf.keras.layers.Dense(10, activation='softmax')
])


print("---------------读取数据并训练模型----------------")
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.5),
             loss = 'sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5,
              batch_size=256,
              validation_data=(x_test, y_test),
              validation_freq=1)






