#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : monte_carlo_accept_reject_sample.py
@Author : jeffsheng
@Date : 2020/2/20 0020
@Desc : 蒙特卡洛思想下的接受拒绝采样
蒙特卡洛思想：精确解析的方法不行的时候，我们就采用大量样本近似的方法去对问题进行近似求解，在大数定理的支撑下，
这种大样本近似方法最终的期望是和精确解是一致的，这就是蒙特卡洛方法的理论支撑
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm
import seaborn
seaborn.set()

# 目标采样分布的概率密度函数
def p(x):
    return (0.3 * np.exp(-(x - 0.3) ** 2) + 0.7 * np.exp(-(x - 2.) ** 2 / 0.3)) / 1.2113

# 建议分布
norm_rv = norm(loc=1.4, scale=1.2)
# C 值
C = 2.5

uniform_rv = uniform(loc=0, scale=1)
sample = []

for i in range(100000):
    Y = norm_rv.rvs(1)[0]
    U = uniform_rv.rvs(1)[0]
    if p(Y) >= U * C * norm_rv.pdf(Y):
        sample.append(Y)

x = np.arange(-3., 5., 0.01)
plt.gca().axes.set_xlim(-3, 5)
plt.plot(x, p(x), color='r')
plt.hist(sample, color='b', bins=150, density=True, edgecolor='k')
plt.show()



