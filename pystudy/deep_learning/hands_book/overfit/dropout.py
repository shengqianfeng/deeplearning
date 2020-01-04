#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : dropout.py
@Author : jeffsheng
@Date : 2020/1/2 0002
@Desc :tensorflow使用丢弃发抑制过拟合
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras, nn, losses
import pystudy.deep_learning.hands_book.d2lzh_pytorch.utils as d2l

def dropout(X, drop_prob):
    """
    :param X:
    :param drop_prob: 丢弃概率
    :return:
    """
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    # 这种情况下把全部元素都丢弃
    if keep_prob == 0:
        return tf.zeros_like(X)
    #初始mask为一个bool型数组，故需要强制类型转换
    # 生成的是形状为X.shape,且[0,1]范围内的均匀分布
    mask = tf.random.uniform(shape=X.shape, minval=0, maxval=1) < keep_prob
    # 将bool值转为浮点数0或者1
    mask = tf.cast(mask, dtype=tf.float32)
    X = tf.cast(X, dtype=tf.float32)
    return mask * X / keep_prob

# 生成一个（2,8）形状的张量
X = tf.reshape(tf.range(0, 16), shape=(2, 8))
"""
全部保留
tf.Tensor(
[[ 0.  1.  2.  3.  4.  5.  6.  7.]
 [ 8.  9. 10. 11. 12. 13. 14. 15.]], shape=(2, 8), dtype=float32)
"""
print(dropout(X, 0))

"""
丢弃一半元素
tf.Tensor(
[[ 0.  0.  4.  6.  0.  0. 12.  0.]
 [ 0.  0.  0. 22. 24. 26. 28. 30.]], shape=(2, 8), dtype=float32)
"""
print(dropout(X, 0.5))

"""
全部丢弃
tf.Tensor(
[[0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0]], shape=(2, 8), dtype=int32)
"""
print(dropout(X, 1.0))


print("------------定义模型参数---------------")

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

W1 = tf.Variable(tf.random.normal(stddev=0.01, shape=(num_inputs, num_hiddens1)))
b1 = tf.Variable(tf.zeros(num_hiddens1))
W2 = tf.Variable(tf.random.normal(stddev=0.1, shape=(num_hiddens1, num_hiddens2)))
b2 = tf.Variable(tf.zeros(num_hiddens2))
W3 = tf.Variable(tf.random.truncated_normal(stddev=0.01, shape=(num_hiddens2, num_outputs)))
b3 = tf.Variable(tf.zeros(num_outputs))

params = [W1, b1, W2, b2, W3, b3]


print("----------------定义模型---------------")
# 把第一个隐藏层的丢弃概率设为0.2，把第二个隐藏层的丢弃概率设为0.5
drop_prob1, drop_prob2 = 0.2, 0.5


def net(X, is_training=False):
    X = tf.reshape(X, shape=(-1,num_inputs))
    H1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    if is_training:# 只在训练模型时使用丢弃法
      H1 = dropout(H1, drop_prob1)  # 在第一层全连接后添加丢弃层
    H2 = nn.relu(tf.matmul(H1, W2) + b2)
    if is_training:
      H2 = dropout(H2, drop_prob2)  # 在第二层全连接后添加丢弃层
    return tf.math.softmax(tf.matmul(H2, W3) + b3)



def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for _, (X, y) in enumerate(data_iter):
        y = tf.cast(y,dtype=tf.int64)
        acc_sum += np.sum(tf.cast(tf.argmax(net(X), axis=1), dtype=tf.int64) == y)
        n += y.shape[0]
    return acc_sum / n

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, trainer=None):
    global sample_grads
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                y_hat = net(X, is_training=True)
                l = tf.reduce_sum(loss(y_hat, tf.one_hot(y, depth=10, axis=-1, dtype=tf.float32)))

            grads = tape.gradient(l, params)
            if trainer is None:

                sample_grads = grads
                params[0].assign_sub(grads[0] * lr)
                params[1].assign_sub(grads[1] * lr)
            else:
                trainer.apply_gradients(zip(grads, params))  # “softmax回归的简洁实现”一节将用到

            y = tf.cast(y, dtype=tf.float32)
            train_l_sum += l.numpy()
            train_acc_sum += tf.reduce_sum(tf.cast(tf.argmax(y_hat, axis=1) == tf.cast(y, dtype=tf.int64), dtype=tf.int64)).numpy()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


print("-----------训练和测试模型----------------")
from tensorflow.keras.datasets import fashion_mnist
batch_size=256
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = tf.cast(x_train, tf.float32) / 255 #在进行矩阵相乘时需要float型，故强制类型转换为float型
x_test = tf.cast(x_test,tf.float32) / 255 #在进行矩阵相乘时需要float型，故强制类型转换为float型
train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)


num_epochs, lr, batch_size = 5, 0.5, 256
loss = tf.losses.CategoricalCrossentropy()
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

