#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : model_param_store.py
@Author : jeffsheng
@Date : 2020/1/3
@Desc : 把内存中训练好的模型参数存储在硬盘上供后续读取使用
"""

import tensorflow as tf
import numpy as np
print(tf.__version__)


x = tf.ones(3)
print(x)

print("--------将数据从存储的文件读回内存--------")
# save函数和load函数分别存储和读取
np.save('x.npy', x)
x2 = np.load('x.npy')
print(x2)


print("------还可以存储一列tensor并读回内存-----------")
y = tf.zeros(4)
np.save('xy.npy',[x,y])
x2, y2 = np.load('xy.npy', allow_pickle=True)
print(x2, y2)


print("----甚至可以存储并读取一个从字符串映射到tensor的字典----------")
mydict = {'x': x, 'y': y}
np.save('mydict.npy', mydict)
mydict2 = np.load('mydict.npy', allow_pickle=True)
print(mydict2)

print("------->>>>>>>>>>>读写模型的参数------------------")
X = tf.random.normal((2,20))
print(X)

class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()    # Flatten层将除第一维（batch_size）以外的维度展平
        self.dense1 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        output = self.dense2(x)
        return output


net = MLP()
Y = net(X)
print(Y)
print("---------把该模型的参数存成文件，文件名为4.5saved_model.h5-----------")
net.save_weights("4.5saved_model.h5")

print("--------直接读取保存在文件里的参数----------------")
net2 = MLP()
net2(X)
net2.load_weights("4.5saved_model.h5")
Y2 = net2(X)
print(Y2 == Y)




