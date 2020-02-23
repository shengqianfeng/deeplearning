#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : second_standard_normal_visual.py
@Author : jeffsheng
@Date : 2020/2/19 0019
@Desc : 二元标准正态分布的可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set()
# 正态分布的均值
mean = np.array([0, 0])
# 生成了各自均值为0，方差为1，随机变量间的协方差为0的二元标准正态分布随机变量X和Y
# 协方差矩阵
conv = np.array([[1, 0],
                 [0, 1]])
# size指定生成矩阵的维度
x, y = np.random.multivariate_normal(mean=mean, cov=conv, size=5000).T
plt.figure(figsize=(6, 6))
plt.plot(x, y, 'ro', alpha=0.2)
plt.gca().axes.set_xlim(-4, 4)
plt.gca().axes.set_ylim(-4, 4)
"""
图像结果显示：原点附近样本点出现的概率最高，非常密集
"""
plt.show()




