#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : second_correlation_coefficient_visual.py
@Author : jeffsheng
@Date : 2020/2/19 0019
@Desc : 问题：协方差大的两个随机变量，他们之间的相关性一定就大于协方差小的随机变量吗？
通过可视化图像来观察协方差的大小对相关性的影响。
结果：
答案是不一定，随机变量的量纲选取的不同，会对方差和协方差的结果值带来数值上的影响，因此对协方差进行标准化
就得到了相关系数的概念:
① 经过标准化处理之后的相关系数，他的取值介于 [−1,1] 之间，相关系数为 0，说明随机变量之间相互独立
② 相关系数的绝对值越接近1，随机变量之间的相关性越强，样本分布图像呈现的椭圆就越窄，如果取到1，图像收缩为一条直线
③ 相关系数为正，随机变量正相关，呈现为右上方倾斜，为负则随机变量负相关，呈现左下方倾斜。
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set()

fig, ax = plt.subplots(1, 2)
mean = np.array([0,0])


conv = np.array([[1, 0.85],
                 [0.85, 1]])
# 正态分布，协方差为0.85
x_1, y_1 = np.random.multivariate_normal(mean=mean, cov=conv, size=3000).T
x_2 = x_1*100
y_2 = y_1*100

ax[0].plot(x_1, y_1, 'bo', alpha=0.05)
ax[1].plot(x_2, y_2, 'bo', alpha=0.05)

S_1 = np.vstack((x_1, y_1))
S_2 = np.vstack((x_2, y_2))
print(np.cov(S_1))
print(np.cov(S_2))

plt.show()


