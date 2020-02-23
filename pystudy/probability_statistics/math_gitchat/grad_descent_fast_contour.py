#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : grad_descent_fast_contour.py
@Author : jeffsheng
@Date : 2020/2/15 0015
@Desc : 最速下降法的等高线可视化
图中的规律：最速下降法的必然结果是相邻两次的迭代搜索方向保持相互间的垂直关系，
即相邻两次的搜索方向满足相互正交，结论可推导，见：
-----gitchat专刊 机器学习中的数学：微积分与最优化10/11多元函数的极值（中）：最速下降法
"""
from sympy import *
from matplotlib import pyplot as plt
import numpy as np


def get_func_val(f, p):
    return f.subs(x1, p[0]).subs(x2, p[1])


def grad_l2(grad_cur, p_cur):
    return sqrt(get_func_val(grad_cur[0], p_cur) ** 2 +
        get_func_val(grad_cur[1], p_cur) ** 2)


def get_alpha(f):
    alpha_list = np.array(solve(diff(f)))
    return min(alpha_list[alpha_list > 0])


def func(x1, x2):
    return 2 * x1 ** 2 + x2 ** 2 - x1 * x2 - 2 * x2


x1 = np.arange(-0.2, 1.2, 0.01)
x2 = np.arange(-0.2, 1.2, 0.01)
x1, x2 = np.meshgrid(x1, x2)

C = plt.contour(x1, x2, func(x1, x2), 60)
plt.clabel(C, inline=True, fontsize=12)

x1 = symbols("x1")
x2 = symbols("x2")
f = 2 * x1 ** 2 + x2 ** 2 - x1 * x2 - 2 * x2

p0 = np.array([0, 0])
p_cur = p0
grad_cur = np.array([diff(f, x1), diff(f, x2)])

while (True):
    plt.plot(float(p_cur[0]), float(p_cur[1]),'ro', markersize=4)
    if (grad_l2(grad_cur, p_cur) < 0.001):
        break
    grad_cur_val = np.array([get_func_val(grad_cur[0], p_cur), get_func_val(grad_cur[1], p_cur)])
    a = symbols('a')
    p_val = p_cur - a * grad_cur_val
    alpha = get_alpha(f.subs(x1, p_val[0]).subs(x2, p_val[1]))
    p_cur = p_cur - alpha * grad_cur_val

plt.show()




