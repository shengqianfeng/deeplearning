#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pystudy.nn_study.d2lzh as d2l
from mxnet import autograd, nd
from mxnet.gluon import data as gdata
import numpy as np
import time
"""
@File : mini_batch_sgd.py
@Author : jeffsheng
@Date : 2019/11/29
@Desc : 小批量随机梯度下降的研究
"""
"""
使用该数据集的前1,500个样本和5个特征，并使用标准化对数据进行预处理
"""
# 使用一个来自NASA的测试不同飞机机翼噪音的数据集airfoil_self_noise.dat
def get_data_ch7():  # 本函数已保存在d2lzh包中方便以后使用
    data = np.genfromtxt('../data/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return nd.array(data[:1500, :-1]), nd.array(data[:1500, -1])

features, labels = get_data_ch7()
print(features.shape)

# 添加了一个状态输入states并将超参数放在字典hyperparams里
# 根据学习率来更新params参数（权重和偏置）
def sgd(params, states, hyperparams):
    for p in params:
        p[:] -= hyperparams['lr'] * p.grad


# 实现一个通用的训练函数
# 它初始化一个线性回归模型，然后可以使用小批量随机梯度下降来训练模型
# 本函数已保存在d2lzh包中方便以后使用
def train_ch7(trainer_fn, states, hyperparams, features, labels,
              batch_size=10, num_epochs=2):
    # 初始化模型
    net, loss = d2l.linreg, d2l.squared_loss
    w = nd.random.normal(scale=0.01, shape=(features.shape[1], 1))
    b = nd.zeros(1)
    w.attach_grad()
    b.attach_grad()

    def eval_loss():
        return loss(net(features, w, b), labels).mean().asscalar()

    ls = [eval_loss()]
    data_iter = gdata.DataLoader(
        gdata.ArrayDataset(features, labels), batch_size, shuffle=True)
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            with autograd.record():
                l = loss(net(X, w, b), y).mean()  # 使用平均损失
            l.backward()
            trainer_fn([w, b], states, hyperparams)  # 迭代模型参数
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    # 打印结果和作图
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    d2l.set_figsize()
    d2l.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('loss')
    d2l.plt.show()

def train_sgd(lr, batch_size, num_epochs=2):
    train_ch7(sgd, None, {'lr': lr}, features, labels, batch_size, num_epochs)

"""
# 学习率为1 minibatch大小1500个样本也就是总样本数 6次迭代
每个迭代周期对样本只迭代一次
"""
print("--------梯度下降演示（minibatch=1500-->样本总数） by jeffsheng 20191201-------------------")
# train_sgd(1, 1500, 6)


print("--------随机梯度下降演示（minibatch=1） by jeffsheng 20191201-------------------")
# 当批量大小为1时，优化使用的是随机梯度下降
"""
为了简化实现，有关（小批量）随机梯度下降的实验中，我们未对学习率进行自我衰减，
而是直接采用较小的常数学习率。随机梯度下降中，每处理一个样本会更新一次自变量（模型参数），
一个迭代周期里会对自变量进行1,500次更新。可以看到，目标函数值的下降在1个迭代周期后就变得较为平缓

虽然随机梯度下降和梯度下降在一个迭代周期里都处理了1,500个样本，但实验中随机梯度下降的一个迭代周期耗时更多。
这是因为随机梯度下降在一个迭代周期里做了更多次的自变量迭代，而且单样本的梯度计算难以有效利用矢量计算
"""
# train_sgd(0.005, 1)

# 当批量大小为10时，优化使用的是小批量随机梯度下降。它在每个迭代周期的耗时介于梯度下降和随机梯度下降的耗时之间
train_sgd(0.05, 10)

