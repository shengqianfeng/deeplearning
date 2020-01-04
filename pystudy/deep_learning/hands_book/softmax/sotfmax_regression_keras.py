#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : sotfmax_regression_keras.py
@Author : jeffsheng
@Date : 2020/1/2
@Desc : softmax回归的简洁实现keras
"""
import tensorflow as tf
from tensorflow import keras

print("--------获取和读取数据------")
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print("------数据归一化------------")
x_train = x_train / 255.0
x_test = x_test / 255.0


print("----------定义和初始化模型----------")
"""
添加一个输出个数为10的全连接层。 
第一层是Flatten，将28 * 28的像素值，压缩成一行 (784, ) 
第二层还是Dense
因为是多分类问题，激活函数使用softmax
"""
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

print("----------softmax和交叉熵损失函数------")
# Tensorflow2.0的keras API提供了一个loss参数。它的数值稳定性更好
loss = 'sparse_categorical_crossentropy'

print("-----------定义优化算法------------------")
optimizer = tf.keras.optimizers.SGD(0.1)


print("----------------训练模型----------------")
model.compile(optimizer=tf.keras.optimizers.SGD(0.1),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5,batch_size=256)
print("---------------比较模型在测试数据集上的表现情况----")
test_loss, test_acc = model.evaluate(x_test, y_test)
# Test Acc: 0.8226
print('Test Acc:',test_acc)










