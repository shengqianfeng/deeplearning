#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : second_noraml_visual.py
@Author : jeffsheng
@Date : 2020/2/19 0019
@Desc : 改变标准正态分布的方差，来转化并观察二元一般正态分布的图像
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set()

mean = np.array([0, 0])
conv_1 = np.array([[1, 0],
                 [0, 1]])

# 对角线上的是随机变量关于自身协方差，也就是方差，此时不是1，也就是非标准正态分布
# 非对角线上的是随机变量之间的协方差为0，说明二者不相关
conv_2 = np.array([[4, 0],
                 [0, 0.25]])

x_1, y_1 = np.random.multivariate_normal(mean=mean, cov=conv_1, size=3000).T
x_2, y_2 = np.random.multivariate_normal(mean=mean, cov=conv_2, size=3000).T
plt.plot(x_1, y_1, 'ro', alpha=0.05)
plt.plot(x_2, y_2, 'bo', alpha=0.05)

plt.gca().axes.set_xlim(-6, 6)
plt.gca().axes.set_ylim(-6, 6)
plt.show()




