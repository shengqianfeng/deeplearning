#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : bayesian_inference_beta_visual.py
@Author : jeffsheng
@Date : 2020/2/20 0020
@Desc : 贝叶斯统计推断中先验分布使用不同α和β的beta分布可视化
通过9种不同的α和β组合，可以看到beta分布的形状出现类似U型分布、正态分布、均匀分布、指数分布等不同的形状


"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import seaborn
seaborn.set()
# 这里选择的α和β是params数组的9组自由组合
params = [0.25, 1, 10]
x = np.linspace(0, 1, 100)
f, ax = plt.subplots(len(params), len(params), sharex=True, sharey=True)

for i in range(len(params)):
    for j in range(len(params)):
        a = params[i]
        b = params[j]
        y = beta(a, b).pdf(x)
        ax[i, j].plot(x, y, color='red')
        ax[i, j].set_title('`$\\alpha$={},$\\beta={}$`'.format(a, b))
        ax[i, j].set_ylim(0, 10)

ax[0, 0].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax[0, 0].set_yticks([0, 2.5, 5, 7.5,  10])
ax[1, 0].set_ylabel('`$p(\\theta)$`')
ax[2, 1].set_xlabel('`$\\theta$`')
plt.show()




