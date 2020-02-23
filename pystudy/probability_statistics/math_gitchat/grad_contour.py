#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : grad_contour.py
@Author : jeffsheng
@Date : 2020/2/14 0014
@Desc :验证等位线和梯度向量的垂直关系
    随机在定义域内选取三个点 (-1.5,1.5)、(-1.5,-1)、(1.0,0)，并绘制其梯度向量，观察是否满足和对应点处等位线的垂直关系
"""

import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return y**2-x**2

def grad_x(f, x, y):
    h = 1e-4
    return (f(x + h/2, y) - f(x - h/2, y)) / h

def grad_y(f, x, y):
    h = 1e-4
    return (f(x, y + h/2) - f(x, y - h/2)) / h

x = np.arange(-2, 2, 0.01)
y = np.arange(-2, 2, 0.01)
X, Y = np.meshgrid(x, y)

#添加等高线
C = plt.contour(X, Y, f(X, Y), 36)
#增加各等高线的高度值
plt.clabel(C, inline=True, fontsize=12)

plt.quiver(-1.5, -1, grad_x(f, -1.5, -1), grad_y(f, -1.5, -1))
plt.quiver(1.0, 0, grad_x(f, 1.0, 0), grad_y(f, 1.0, 0))
plt.quiver(-1.5, 1.5, grad_x(f, -1.5, 1.5), grad_y(f, -1.5, 1.5))
plt.grid()
plt.show()



