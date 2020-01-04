#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pystudy.nn_study.d2lzh as d2l
import numpy as np
"""
@File : multi_func_gradient_descent.py
@Author : jeffsheng
@Date : 2019/11/29
@Desc : 多维函数梯度下降
"""

# 构造一个输入为二维向量 x=[x1,x2]⊤ 和输出为标量的目标函数 f(x)=x1^2+2*x2^2 。
# 那么，梯度 ∇f(x)=[2*x1,4*x2]⊤ 。我们将观察梯度下降从初始位置 [−5,−2]开始对自变量 xx 的迭代轨迹。
# 我们先定义两个辅助函数，第一个函数使用给定的自变量更新函数，从初始位置 [−5,−2]开始迭代自变量 x 共20次，
# 第二个函数对自变量 x 的迭代轨迹进行可视化

def train_2d(trainer):  # 本函数将保存在d2lzh包中方便以后使用
    x1, x2, s1, s2 = -5, -2, 0, 0  # s1和s2是自变量状态，本章后续几节会使用
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print('epoch %d, x1 %f, x2 %f' % (i + 1, x1, x2))
    return results

def show_trace_2d(f, results):  # 本函数将保存在d2lzh包中方便以后使用
    # 在函数曲线上画点(x1,x2) 橙色，这些点是对x1和x2同时分别求导20次的梯度值
    # *zip(*results) x1和x2两个集合
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    # x1和x2 确定定义域的范围
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    # 绘制x1和x2的取值等高线，蓝色
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
    d2l.plt.show()

# 观察学习率为 0.1时自变量的迭代轨迹,使用梯度下降对自变量 x迭代20次后，可见最终 x的值较接近最优解 [0,0]
eta = 0.1


def f_2d(x1, x2):  # 目标函数
    return x1 ** 2 + 2 * x2 ** 2


def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 2 * x1, x2 - eta * 4 * x2, 0, 0)

show_trace_2d(f_2d, train_2d(gd_2d))

print("--------------------初识随机梯度下降---------------------------")
"""
随机梯度下降说明：
    1 梯度下降是拿整体样本总数n，则n个损失函数的平均值作为每次迭代的结果，时间复杂度每次迭代O(n),所以
    梯度下降有时也叫批量梯度下降。
    2 随机梯度下降是每次迭代随机从n个样本中挑出一个样本，一个样本的损失函数值作为每次迭代结果，时间复杂度O(1)
《动手深度学习》7.2 by jeffsheng 20191201     
    3 小批量随机梯度下降是在每次迭代中随机均匀采样多个样本组成一个小批量，随后使用这个小批量来计算梯度。每次迭代
    的时间复杂度O(mini batch size)
    
"""
# 通过在梯度中添加均值为0的随机噪声来模拟随机梯度下降，以此来比较它与梯度下降的区别
def sgd_2d(x1, x2, s1, s2):
    return (x1 - eta * (2 * x1 + np.random.normal(0.1)),
            x2 - eta * (4 * x2 + np.random.normal(0.1)), 0, 0)
# 随机梯度下降中自变量的迭代轨迹相对于梯度下降中的来说更为曲折。这是由于实验所添加的噪声使模拟的随机梯度的准确度下降。
# 在实际中，这些噪声通常指训练数据集中的无意义的干扰
# show_trace_2d(f_2d, train_2d(sgd_2d))