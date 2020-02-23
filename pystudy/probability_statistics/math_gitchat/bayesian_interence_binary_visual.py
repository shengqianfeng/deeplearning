#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : bayesian_interence_binary_visual.py
@Author : jeffsheng
@Date : 2020/2/20 0020
@Desc : 贝叶斯统计推断中先验事件（比如抛掷硬币正面发生，正面发生概率设为θ）发生的条件下，
观测数据x发生（比如x次正面）的概率p(x|θ)符合二项分布，以下选择三种不同正面发生概率：
0.35,0.5,0.8抛掷10次硬币的试验图像。

贝叶斯推断的核心思想：
    先验分布+观测数据=后验分布
"""

from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np
import seaborn

seaborn.set()
# 抛掷硬币次数为10
n = 10
# 正面向上的概率分为三组进行试验
p_params = [0.35, 0.5, 0.8]
x = np.arange(0, n + 1)
f, ax = plt.subplots(len(p_params), 1)

for i in range(len(p_params)):
    p = p_params[i]
    # 二项分布
    y = binom(n=n, p=p).pmf(x)

    ax[i].vlines(x, 0, y, colors='red', lw=10)
    ax[i].set_ylim(0, 0.5)
    ax[i].plot(0, 0, label='n={}\n`$\\theta$`={}'.format(n, p), alpha=0)
    ax[i].legend()
    ax[i].set_xlabel('y')
    ax[i].set_xticks(x)

ax[1].set_ylabel('`$p(y|\\theta)$`')
plt.show()



