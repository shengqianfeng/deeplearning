#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : newton_super_fast.py
@Author : jeffsheng
@Date : 2020/2/15 0015
@Desc : 牛顿法求多元函数的极值（了解）
这是最速下降法的改进版，参考资料是gitchat张雨萌高数专栏最后一讲《牛顿法与向量微分》
"""
from sympy import *
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)


def func(x1,x2):
    return 2*x1**2+x2**2-x1*x2-2*x2

x1 = np.arange(-1.5, 1.5, 0.01)
x2 = np.arange(-1.5, 1.5, 0.01)
x1, x2 = np.meshgrid(x1, x2)
ax.plot_surface(x1, x2, func(x1, x2), color='y', alpha=0.3)

x1 = symbols("x1")
x2 = symbols("x2")
f = 2*x1**2+x2**2-x1*x2-2*x2
p0 = np.array([0, 0], dtype=float)
p_cur = p0
max_iter = 10000

for i in range(max_iter):
    grad_cur = np.array([diff(f, x1).subs(x1, p_cur[0]).subs(x2, p_cur[1]),
                         diff(f, x2).subs(x1, p_cur[0]).subs(x2, p_cur[1])], dtype=float)

    ax.scatter(float(p_cur[0]), float(p_cur[1]), float(f.subs(x1,p_cur[0]).subs(x2, p_cur[1])), color='r')
    print(p_cur)
    if np.linalg.norm(grad_cur, ord=2) < 0.0001:
        break
    hessian_M = np.array([[diff(f, x1, 2).subs(x1, p_cur[0]).subs(x2, p_cur[1]),
                           diff(f, x1, 1, x2, 1).subs(x1, p_cur[0]).subs(x2, p_cur[1])],
                          [diff(f, x2, 1, x1, 1).subs(x1, p_cur[0]).subs(x2, p_cur[1]),
                           diff(f, x2, 2).subs(x1, p_cur[0]).subs(x2, p_cur[1])]],
                         dtype=float)  # 黑塞矩阵

    hessian_inv = np.linalg.inv(hessian_M)  # 黑塞矩阵求逆
    p_cur = p_cur - np.dot(hessian_inv, grad_cur)

plt.show()




