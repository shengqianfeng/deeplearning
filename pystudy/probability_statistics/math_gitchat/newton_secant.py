#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : newton_secant.py
@Author : jeffsheng
@Date : 2020/2/15 0015
@Desc : 割线法求极值（二阶导数不存在的情况使用）
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn
seaborn.set()

def f(x):
    return (1/4)*x**4-4*x**3+4*x**2+40*x+5


def df(x):
    return x**3-12*x**2+8*x+40

def secant(xk,xk_1):
    return xk-df(xk)*(xk-xk_1)/(df(xk)-df(xk_1))

x0 = 13
x1 = 12
x_list = []
x_list.append(x0)
x_list.append(x1)

while(True):
    x = secant(x_list[-1],x_list[-2])
    if abs(x-x_list[-1]) <= 1e-5:
        break
    x_list.append(x)


fig, ax = plt.subplots(2, 1)

x = np.linspace(0, 20, 1000)
ax[0].plot(x, f(x))

x = np.linspace(7.5, 15, 1000)
ax[1].plot(x, f(x), label='f(x)')
ax[1].plot(x, df(x), label='df(x)')
ax[1].plot(x_list,[0]*len(x_list),'ko')
ax[1].legend()
plt.show()

print(x_list)
print('x={},f(x)={},df(x)={}'.format(x_list[-1],f(x_list[-1]),df(x_list[-1])))

