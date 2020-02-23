#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : central_limit_theorem.py
@Author : jeffsheng
@Date : 2020/2/19 0019
@Desc : 中心极限定理的模拟
结果：随着单次采样个数的逐渐增加，标准化之后的随机变量越来越像标准正态分布了
"""
import numpy as np
from scipy.stats import geom
import matplotlib.pyplot as plt
import seaborn
seaborn.set()

fig, ax = plt.subplots(2, 2)
# 原始分布为参数为0.3的几何分布
geom_rv = geom(p=0.3)
geom_rvs = geom_rv.rvs(size=1000000)
mean, var, skew, kurt = geom_rv.stats(moments='mvsk')
ax[0, 0].hist(geom_rvs, bins=100, density=True)
ax[0, 0].set_title('geom distribution:p=0.3')
# 采样的样本数列表
n_array = [0, 2, 5, 50]

# 三组试验
for i in range(1, 4):
    Z_array = []
    n = n_array[i]
    # 重复100000次【采样+标注化】步骤
    for j in range(100000):
        # n个样本
        sample = np.random.choice(geom_rvs, n)
        # 标准化
        Z_array.append((sum(sample) - n * mean) / np.sqrt(n * var))
    ax[i//2, i%2].hist(Z_array, bins=100, density=True)
    ax[i//2, i%2].set_title('n={}'.format(n))
    ax[i//2, i%2].set_xlim(-3, 3)

plt.show()




