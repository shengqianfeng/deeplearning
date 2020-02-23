#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : grad_field.py
@Author : jeffsheng
@Date : 2020/2/14 0014
@Desc : 可视化二元函数的梯度场
梯度场的概念：
    对于二元函数来说，梯度就是(fx,fy),二元函数 f(x,y)，在它的定义域内，如果我们把每一个点的梯度都求出来，
将每个点的梯度向量和各个点的位置联系起来进行集中展示，就形成了一个梯度场
"""

import numpy as np
import matplotlib.pyplot as plt

# 原函数
def f(x, y):
    return x**2-y**2

# 对x求偏导数
def grad_x(f, x, y):
    h = 1e-4
    return (f(x + h/2, y) - f(x - h/2, y)) / h

# 对y求偏导数
def grad_y(f, x, y):
    h = 1e-4
    return (f(x, y + h/2) - f(x, y - h/2)) / h

def numerical_gradient(f,P):
    """
    求取整个定义域上的梯度
    :param f:梯度函数
    :param P: P为 2×256的数组
    P[0]:就是所有点的 x 坐标构成的一维数组 X,P[1] 就是所有点的 y 坐标构成的二维数组 Y
    :return:
    """
    grad = np.zeros_like(P)
    # 其中 X[i] 和 Y[i] 分别对应表示这 256 个点中第 i 个点的横纵坐标
    for i in range(P[0].size):
        grad[0][i] = grad_x(f, P[0][i], P[1][i])
        grad[1][i] = grad_y(f, P[0][i], P[1][i])
    return grad


# x和y分别形成了16 个点
x = np.arange(-2, 2, 0.25)
y = np.arange(-2, 2, 0.25)

# 将x和y进行网格化，得到的X和Y是一个16*16的二维数组，X代表了(x,y)平面上的256个点的横坐标，同理Y代表了（x,y）平面上的纵坐标
X, Y = np.meshgrid(x, y)
# 这两行代码将这两个二维数组展平成一维数组，X 和 Y 都变成含有 256 个元素的一维数组
X = X.flatten()
Y = Y.flatten()

grad = numerical_gradient(f, np.array([X, Y]))

# 将所有点的梯度用箭头的形式绘制出来,X,Y表示箭头的起点， grad[0], grad[1]表示箭头的方向
plt.quiver(X, Y, grad[0], grad[1])#grad[0]是一个1*X.size的数组
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()


