#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : grad_descent_fast.py
@Author : jeffsheng
@Date : 2020/2/15 0015
@Desc : 最速下降法的实现
"""

from sympy import *
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)

# 将p(x1,x2)代入函数中求极小值
def get_func_val(f, p):
    return f.subs(x1, p[0]).subs(x2, p[1])


# 计算当前点处梯度向量的模长
def grad_l2(grad_cur, p_cur):
    return sqrt(get_func_val(grad_cur[0], p_cur) ** 2 +
                get_func_val(grad_cur[1], p_cur) ** 2)

# 解方程得到最终函数的解集合中的最小值就是极小值
def get_alpha(f):
    alpha_list = np.array(solve(diff(f)))
    return min(alpha_list[alpha_list>0])

# 原函数
def func(x1,x2):
    return 2*x1**2+x2**2-x1*x2-2*x2


x1 = np.arange(-1.5, 1.5, 0.01)
x2 = np.arange(-1.5, 1.5, 0.01)
x1, x2 = np.meshgrid(x1, x2)
ax.plot_surface(x1, x2, func(x1, x2), color='y', alpha=0.3)

# 定义自变量x1和x2以及二元函数表达式
x1 = symbols("x1")
x2 = symbols("x2")
f = 2*x1**2+x2**2-x1*x2-2*x2

p0 = np.array([0, 0])
p_cur = p0
grad_cur = np.array([diff(f, x1), diff(f, x2)])

while(True):
    ax.scatter(float(p_cur[0]),float(p_cur[1]),func(float(p_cur[0]),float(p_cur[1])),color='r')
    if (grad_l2(grad_cur, p_cur) < 0.001):
        break
    # 梯度向量（x1的偏导，x2的偏导）
    grad_cur_val = np.array([get_func_val(grad_cur[0], p_cur),get_func_val(grad_cur[1], p_cur)])
    a = symbols('a')
    p_val = p_cur - a * grad_cur_val
    alpha = get_alpha(f.subs(x1, p_val[0]).subs(x2, p_val[1]))
    p_cur = p_cur - alpha * grad_cur_val

plt.show()




