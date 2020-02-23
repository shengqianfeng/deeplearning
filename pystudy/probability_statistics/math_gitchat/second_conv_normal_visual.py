#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : second_conv_normal_visual.py
@Author : jeffsheng
@Date : 2020/2/19 0019
@Desc : 方差保持不变，通过改变随机变量之间的协方差来观察随机变量之间的相关性
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set()

fig, ax = plt.subplots(2, 2)
mean = np.array([0,0])

conv_1 = np.array([[1, 0],
                 [0, 1]])

# 以下为三组不同的协方差
conv_2 = np.array([[1, 0.3],
                 [0.3, 1]])

conv_3 = np.array([[1, 0.85],
                 [0.85, 1]])

conv_4 = np.array([[1, -0.85],
                 [-0.85, 1]])

# 可以从图像观察到，协方差为越大椭圆图像越窄
x_1, y_1 = np.random.multivariate_normal(mean=mean, cov=conv_1, size=3000).T
x_2, y_2 = np.random.multivariate_normal(mean=mean, cov=conv_2, size=3000).T
x_3, y_3 = np.random.multivariate_normal(mean=mean, cov=conv_3, size=3000).T
x_4, y_4 = np.random.multivariate_normal(mean=mean, cov=conv_4, size=3000).T

ax[0][0].plot(x_1, y_1, 'bo', alpha=0.05)
ax[0][1].plot(x_2, y_2, 'bo', alpha=0.05)
ax[1][0].plot(x_3, y_3, 'bo', alpha=0.05)
ax[1][1].plot(x_4, y_4, 'bo', alpha=0.05)

plt.show()




