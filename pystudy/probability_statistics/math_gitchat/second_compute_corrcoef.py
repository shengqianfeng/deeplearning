#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : second_compute_corrcoef.py
@Author : jeffsheng
@Date : 2020/2/19 0019
@Desc : 计算两组随机变量之间的相关系数
比较随机变量之间的相关性，只看一个指标：那就是相关系数（而不是盯着协方差的取值），
相关系数去除了不同量纲所带来的影响。相关系数的绝对值越大，相关性越强
"""


import numpy as np

mean = np.array([0,0])
conv = np.array([[1, 0.85],
                 [0.85, 1]])

x_1, y_1 = np.random.multivariate_normal(mean=mean, cov=conv, size=3000).T
x_2 = x_1*100
y_2 = y_1*100

S_1 = np.vstack((x_1, y_1))
S_2 = np.vstack((x_2, y_2))

# 结果是协方差矩阵
print(np.corrcoef(S_1))
print(np.corrcoef(S_2))

