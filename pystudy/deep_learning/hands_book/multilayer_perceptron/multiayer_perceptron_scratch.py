#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : multiayer_perceptron_scratch.py
@Author : jeffsheng
@Date : 2020/1/2
@Desc : tensorflow多层感知机从0开始实现
"""
import tensorflow as tf
import numpy as np
import sys
print(tf.__version__)

print("--------------获取和读取数据-------------------")
from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
batch_size = 256
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)
x_train = x_train/255.0
x_test = x_test/255.0
train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)


print("----------定义模型参数-------------------")
# 定义隐藏层神经元个数为256
num_inputs, num_outputs, num_hiddens = 784, 10, 256

w1 = tf.Variable(tf.random.truncated_normal([num_inputs, num_hiddens], stddev=0.1))
b1 = tf.Variable(tf.random.truncated_normal([num_hiddens], stddev=0.1))
w2 = tf.Variable(tf.random.truncated_normal([num_hiddens, num_outputs], stddev=0.1))
b2=tf.Variable(tf.random.truncated_normal([num_outputs], stddev=0.1))


print("---------定义激活函数----------------")
def relu(x):
    return tf.math.maximum(x,0)


print("--------------定义模型---------------")
def net(x,w1,b1,w2,b2):
    x = tf.reshape(x,shape=[-1,num_inputs])
    h = relu(tf.matmul(x,w1) + b1 )
    y = tf.math.softmax( tf.matmul(h,w2) + b2 )
    return y


print("--------------定义损失函数-------------")


def loss(y_hat,y_true):
    # 直接使用Tensorflow提供的包括softmax运算和交叉熵损失计算的函数
    return tf.losses.sparse_categorical_crossentropy(y_true,y_hat)



def acc(y_hat,y):
    return np.mean((tf.argmax(y_hat,axis=1) == y))


print("------------训练模型---------------------")
num_epochs, lr = 5, 0.5
for epoch in range(num_epochs):
    loss_all = 0
    for x,y in train_iter:
        with tf.GradientTape() as tape:
            y_hat = net(x,w1,b1,w2,b2)
            l = tf.reduce_mean(loss(y_hat,y))
            loss_all += l.numpy()
            grads = tape.gradient(l, [w1, b1, w2, b2])
            w1.assign_sub(grads[0])
            b1.assign_sub(grads[1])
            w2.assign_sub(grads[2])
            b2.assign_sub(grads[3])
    print(epoch, 'loss:', l.numpy())
    total_correct, total_number = 0, 0

    for x,y in test_iter:
        with tf.GradientTape() as tape:
            y_hat = net(x,w1,b1,w2,b2)
            y=tf.cast(y,'int64')
            correct=acc(y_hat,y)
    print(epoch,"test_acc:", correct)













