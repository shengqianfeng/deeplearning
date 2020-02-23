#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : newton_tangent.py
@Author : jeffsheng
@Date : 2020/2/15 0015
@Desc : 牛顿法求函数的根
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn
seaborn.set()

# 原函数
def f(x):
    return x**3-12*x**2+8*x+40


# 原函数的一阶导数
def df(x):
    return 3*x**2-24*x+8

# 牛顿法的切线公式（由一阶泰勒公式近似得到）
def newton(x):
    return x - f(x)/df(x)

# 牛顿迭代法的初始值
x0 = 14
x_list = []
x_list.append(x0)
while(True):
    x = newton(x_list[-1])
    if abs(x-x_list[-1])<=1e-5:
        break
    x_list.append(x)

fig, ax = plt.subplots(2, 1)

x = np.linspace(0, 15, 1000)
plt.xlim(0, 15)
ax[0].plot(x, f(x))

x = np.linspace(10.5, 14.5, 1000)
plt.xlim(10.5, 14.5)
ax[1].plot(x, f(x))
ax[1].plot(x_list, [0]*len(x_list), 'ko')
# 打印最终的根x值及对应的函数值已经很接近0了
print('x={},f(x)={}'.format(x_list[-1], f(x_list[-1])))
print(x_list)
plt.show()

