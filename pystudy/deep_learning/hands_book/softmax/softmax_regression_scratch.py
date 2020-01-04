#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : softmax_regression_scratch.py
@Author : jeffsheng
@Date : 2020/1/2
@Desc : softmax从0开始实现
"""

import tensorflow as tf
import numpy as np
print(tf.__version__)

print("------- 获取和读取数据--------")
from tensorflow.keras.datasets import fashion_mnist

batch_size=256
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# 在进行矩阵相乘时需要float型，故强制类型转换为float型
x_train = tf.cast(x_train, tf.float32) / 255
# 在进行矩阵相乘时需要float型，故强制类型转换为float型
x_test = tf.cast(x_test,tf.float32) / 255
train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

print("----------初始化模型参数----------------")
num_inputs = 784
num_outputs = 10
# 使用均值为0，标准差为0.01的正态分布来初始化权重，类型为浮点数
W = tf.Variable(tf.random.normal(shape=(num_inputs, num_outputs), mean=0, stddev=0.01, dtype=tf.float32))
b = tf.Variable(tf.zeros(num_outputs, dtype=tf.float32))

print("-----------实现softmax运算----------")
X = tf.constant([[1, 2, 3], [4, 5, 6]])
"""
tf.reduce_sum函数介绍：
    给定一个Tensor矩阵X,对其中同一列（axis=0）或同一行（axis=1）的元素求和。
    keepdims=True:在结果中保留行和列这两个维度

结果：
tf.Tensor([[5 7 9]], shape=(1, 3), dtype=int32) 
tf.Tensor([[ 6] [15]], shape=(2, 1), dtype=int32)
"""
print(tf.reduce_sum(X, axis=0, keepdims=True), tf.reduce_sum(X, axis=1, keepdims=True))

"""
softmax运算的输出矩阵中的任意一行元素代表了一个样本在各个输出类别上的预测概率.
运算原理：
    为了表达样本预测各个输出的概率，softmax运算会先通过exp函数对每个元素做指数运算，
    再对exp矩阵同行元素求和，最后令矩阵每行各元素与该行元素之和相除。
    这样一来，最终得到的矩阵每行元素和为1且非负。
    因此，该矩阵每行都是合法的概率分布。
"""
def softmax(logits, axis=-1):
    """
    :param logits:  矩阵logits的行数是样本数，列数是输出个数.比如在mnist分类中logits一般是（1，10）
    :param axis:
    :return:
    """
    return tf.exp(logits)/tf.reduce_sum(tf.exp(logits), axis, keepdims=True)

# 对于随机输入，我们将每个元素变成了非负数，且每一行和为1
X = tf.random.normal(shape=(2, 5))
X_prob = softmax(X)
"""
tf.Tensor(
[[0.09837885 0.25165763 0.12309176 0.13924469 0.38762704]
 [0.19698197 0.13644852 0.02483969 0.139318   0.50241184]], shape=(2, 5), dtype=float32) 

tf.Tensor([1. 1.], shape=(2,), dtype=float32)
"""
print(X_prob, tf.reduce_sum(X_prob, axis=1))


print("--------定义模型-------------")
def net(X):
    logits = tf.matmul(tf.reshape(X, shape=(-1, W.shape[0])), W) + b
    return softmax(logits)


print("------------定义损失函数-----------------")
y_hat = np.array([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = np.array([0, 2], dtype='int32')
"""
tf.Tensor(
[[1. 0. 0.]
 [0. 0. 1.]], shape=(2, 3), dtype=float32)
"""
print(tf.one_hot(y, depth=3))
"""
# tf.Tensor([0.1 0.5], shape=(2,), dtype=float64)
# boolean_mask(a,b) 将使a矩阵仅保留与b中“True”元素同下标的部分
"""
print(tf.boolean_mask(y_hat, tf.one_hot(y, depth=3)))

# 交叉熵损失函数 输入y为一维向量(256,)，y_hat为（256,10）
def cross_entropy(y_hat, y):
    y = tf.cast(tf.reshape(y, shape=[-1, 1]),dtype=tf.int32)
    y = tf.one_hot(y, depth=y_hat.shape[-1])
    y = tf.cast(tf.reshape(y, shape=[-1, y_hat.shape[-1]]),dtype=tf.int32) # 将y最终转化为跟y_hat一样的形状这里为（256,10）
    return -tf.math.log(tf.boolean_mask(y_hat, y)+1e-8) # 计算得到y_hat中每一行的最大概率，使用tf.math.log计算(256,)每一个元素的对数,返回（256，）


print("------------计算分类准确率-------------------")
def accuracy(y_hat, y):
    return np.mean((tf.argmax(y_hat, axis=1) == y))


# 0.5
# print(accuracy(y_hat, y))


# 评价模型net在数据集data_iter上的准确率
# 描述,对于tensorflow2中，比较的双方必须类型都是int型，所以要将输出和标签都转为int型
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for _, (X, y) in enumerate(data_iter):
        y = tf.cast(y,dtype=tf.int64)
        acc_sum += np.sum(tf.cast(tf.argmax(net(X), axis=1), dtype=tf.int64) == y)
        n += y.shape[0]
    return acc_sum / n

# 测试随机权重模型的准确率 0.0678
# print(evaluate_accuracy(test_iter, net))


print("---------训练模型----------------")
# 这里使用 1e-3 学习率，是因为原文 0.1 的学习率过大，会使 cross_entropy loss 计算返回 numpy.nan
num_epochs, lr = 5, 1e-3


# 本函数已保存在d2lzh包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, trainer=None):
    global sample_grads
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                y_hat = net(X)
                l = tf.reduce_sum(loss(y_hat, y)) # 将loss返回的256个样本损失值相加,就是本批次的损失值之和，无维度默认对所有元素求和，也就是交叉熵损失值

            grads = tape.gradient(l, params) # 根据损失值求权重和偏置的梯度
            if trainer is None:
                # 没有指定优化器则手动计算
                sample_grads = grads
                params[0].assign_sub(grads[0] * lr)
                params[1].assign_sub(grads[1] * lr)
            else:
                trainer.apply_gradients(zip(grads, params))  # “softmax回归的简洁实现”一节将用到

            y = tf.cast(y, dtype=tf.float32)
            train_l_sum += l.numpy() # 累加每一个批次的损失值之和
            train_acc_sum += tf.reduce_sum(tf.cast(tf.argmax(y_hat,    axis=1) == tf.cast(y, dtype=tf.int64), dtype=tf.int64)).numpy()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


trainer = tf.keras.optimizers.SGD(lr)
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)


print("---------------预测分类----------------------")
# print(W,b)
import matplotlib.pyplot as plt
X, y = iter(test_iter).next()

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images, labels):
    # 这⾥的_表示我们忽略（不使⽤）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12)) # 这里注意subplot 和subplots 的区别
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(tf.reshape(img, shape=(28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


true_labels = get_fashion_mnist_labels(y.numpy())
pred_labels = get_fashion_mnist_labels(tf.argmax(net(X), axis=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

# 比较一下它们的真实标签（第一行文本输出）和模型预测结果（第二行文本输出）
show_fashion_mnist(X[0:9], titles[0:9])

