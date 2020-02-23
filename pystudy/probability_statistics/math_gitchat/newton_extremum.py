#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : newton_extremum.py
@Author : jeffsheng
@Date : 2020/2/15 0015
@Desc : 牛顿法求一元函数的极值
思想：一元函数的极值其实就是 f'(x)=0这个方程的值其中之一，那么还是回到了利用牛顿法求解方程根这个问题上来了。
求g(x)=f'(x)的根会用到f(x)的二阶导数，那么前提肯定是二阶导数得存在，就可以用二阶导数这个切线公式去近似求解了，
否则二阶导数不存在就用割线法代替。
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn
seaborn.set()

def f(x):
    return x**2-2*np.sin(x)

def df(x):
    return 2*x-2*np.cos(x)

def d2f(x):
    return 2+2*np.sin(x)

def newton(x):
    return x - df(x)/d2f(x)

x0 = 3
x_list = []
x_list.append(x0)

while(True):
    x = newton(x_list[-1])
    if abs(x-x_list[-1])<=1e-5:
        break
    x_list.append(x)

fig, ax = plt.subplots(2, 1)

x = np.linspace(-20, 20, 1000)
plt.xlim(-20, 20)
ax[0].plot(x, f(x))

x = np.linspace(-3.5, 3.5, 1000)
plt.xlim(-3.5, 3.5)
plt.ylim(-10, 15)
ax[1].plot(x, f(x), label='f(x)')
ax[1].plot(x, df(x), label='df(x)')
ax[1].plot(x_list, [0]*len(x_list), 'ko')
ax[1].legend(loc='best')
print('x={},f(x)={}'.format(x_list[-1], f(x_list[-1])))
print(x_list,)
plt.show()



