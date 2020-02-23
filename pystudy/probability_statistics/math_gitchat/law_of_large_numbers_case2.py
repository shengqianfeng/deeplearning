#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : law_of_large_numbers_case2.py
@Author : jeffsheng
@Date : 2020/2/19 0019
@Desc : 大数定理模拟案例2
图像 1：原始正态分布的样本分布图像，颜色为蓝色。
图像 2：从1000000 个原始正态分布样本中，每次随机选取5 个数，计算它们的均值，重复操作10000 次，观察这10000 个均值的分布，颜色为红色。
图像 3：从1000000 个原始正态分布样本中，每次随机选取50 个数，计算它们的均值，重复操作10000 次，观察这10000 个均值的分布，颜色为绿色
结果：
随着每次选取的样本数量的增多，样本均值分布的图像越来越向期望集中，再一次佐证了大数定理。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn
seaborn.set()

norm_rvs = norm(loc=0, scale=20).rvs(size=1000000)
plt.hist(norm_rvs, density=True, alpha=0.3, color='b', bins=100, label='original')

mean_array = []
for i in range(10000):
    sample = np.random.choice(norm_rvs, size=5, replace=False)
    mean_array.append(np.mean(sample))
plt.hist(mean_array, density=True, alpha=0.3, color='r', bins=100, label='sample size=5')

for i in range(10000):
    sample = np.random.choice(norm_rvs, size=50, replace=False)
    mean_array.append(np.mean(sample))
plt.hist(mean_array, density=True, alpha=0.3, color='g', bins=100, label='sample size=50')

plt.gca().axes.set_xlim(-60, 60)
plt.legend(loc='best')
plt.show()
