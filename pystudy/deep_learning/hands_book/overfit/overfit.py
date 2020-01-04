#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : overfit.py
@Author : jeffsheng
@Date : 2020/1/2
@Desc : 为了理解模型复杂度和训练数据集大小对欠拟合和过拟合的影响，下面我们以多项式函数拟合为例来进行
多项式函数拟合实验
                y=1.2x − 3.4x^2 + 5.6x^3 + 5 + ϵ,
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import torch

print("-------------生成数据集---------")
# 给定特征x，使用以上三阶多项式来生成标签
# 其中噪声项ϵ 服从均值为0、标准差为0.01的正态分布，训练数据集和测试数据集的样本数都设为100
n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
# torch.randn返回一个范围在{0，1）上均匀分布的张量
features = torch.randn((n_train + n_test, 1))
# torch.pow将输入中每个元素用指数表示并返回张量结果
# torch.cat连接给定的形状相同的张量
poly_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), 1)
# 根据三阶多项式计算结果
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]
          + true_w[2] * poly_features[:, 2] + true_b)
# 给结果标签加上噪音
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

# 查看数据集生成的前两个样本
print(features[:2], poly_features[:2], labels[:2])


print("-----------定义训练和测试模型----------------")
# 本函数已保存在d2lzh_pytorch包中方便以后使用
from IPython import display


def use_svg_display():
    """Use svg format to display plot in jupyter"""
    plt.savefig(fname="name", format="svg")



def set_figsize(figsize=(3.5, 2.5)):
    """Set matplotlib figure size."""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    """
    :param x_vals:  x轴表示：轮次
    :param y_vals:  y轴表示：训练集损失值
    :param x_label: x轴表示：epoach
    :param y_label: y轴表示：损失值
    :param x2_vals: x轴表示：轮次
    :param y2_vals: y轴表示：测试集损失值
    :param legend:  标识
    :param figsize:
    :return:
    """
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # 定义作图函数semilogy，y轴使用对数尺度
    plt.semilogy(x_vals, y_vals) # 绘制训练集图像
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()


# 使用均差平方损失函数
num_epochs, loss = 100, tf.losses.MeanSquaredError()


# 定义模型
def fit_and_plot(train_features, test_features, train_labels, test_labels,type):
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1))
    batch_size = min(10, train_labels.shape[0])
    train_features = np.array(train_features)
    test_features = np.array(test_features)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    train_iter = tf.data.Dataset.from_tensor_slices((train_features, train_labels)).batch(batch_size)
    test_iter = tf.data.Dataset.from_tensor_slices((test_features, test_labels)).batch(batch_size)
    train_features = tf.constant(train_features, tf.float32)
    test_features = tf.constant(test_features, tf.float32)
    train_labels = tf.constant(train_labels, tf.float32)
    test_labels = tf.constant(test_labels, tf.float32)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                l = loss(y, net(X))

            grads = tape.gradient(l, net.trainable_variables)
            optimizer.apply_gradients(zip(grads, net.trainable_variables))

        # 每个epoch统计在训练集和测试集上的损失函数
        train_ls.append(loss(train_labels, net(train_features)).numpy().mean())
        test_ls.append(loss(test_labels, net(test_features)).numpy().mean())
    # epoch迭代完后输出最终损失值结果
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss', range(1, num_epochs + 1), test_ls, [type+'train', type+'test'])
    print('weight:', net.get_weights()[0], '\nbias:', net.get_weights()[1])


print("--------三阶多项式函数拟合（正常）---------")
fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:],'one')

print("----------线性函数拟合（欠拟合）--------------")
fit_and_plot(features[:n_train, :], features[n_train:, :], labels[:n_train],
             labels[n_train:],'two')

print("---------训练样本不足(过拟合)-----------")
fit_and_plot(poly_features[0:2, :], poly_features[n_train:, :], labels[0:2],
             labels[n_train:],'three')



