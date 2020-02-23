#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : law_of_large_numbers.py
@Author : jeffsheng
@Date : 2020/2/19 0019
@Desc :大数定理模拟
三组数据各15000个服从（10,0.4）的二项分布的随机变量，观察
随着样本数目的增大，样本均值和实际分布期望之间的关系。
结果：在每一组试验中，随着样本数量的逐渐增大，样本均值都会越来越收敛于随机变量的期望
"""

import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
import seaborn
seaborn.set()

n = 10
p = 0.4
sample_size = 15000
expected_value = n*p
N_samples = range(1, sample_size, 10)

# 15000个样本，每10个逐步递增求从0到目前递增索引index，求均值，并在坐标轴上可视化
for k in range(3):
    binom_rv = binom(n=n, p=p)
    X = binom_rv.rvs(size=sample_size)
    sample_average = [X[:i].mean() for i in N_samples]
    plt.plot(N_samples, sample_average,
             label='average of sample {}'.format(k))

plt.plot(N_samples, expected_value * np.ones_like(sample_average),
         ls='--', label='true expected value:np={}'.format(n*p), c='k')

plt.legend()
plt.show()


