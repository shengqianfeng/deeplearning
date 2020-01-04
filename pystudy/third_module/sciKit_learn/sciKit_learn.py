#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : sciKit_learn.py
@Author : jeffsheng
@Date : 2019/11/7
@Desc : SciKit-Learn 是 Python 的重要机器学习库，它帮我们封装了大量的机器学习算法，比如分类、聚类、回归、降维等。
此外，它还包括了数据变换模块
"""


from sklearn import preprocessing
import numpy as np


"""
Min-max 规范
们可以让原始数据投射到指定的空间 [min, max]，在 SciKit-Learn 里有个函数 MinMaxScaler 是专门做这个的，
它允许我们给定一个最大值与最小值，然后将原数据投射到 [min, max] 中。
默认情况下 [min,max] 是 [0,1]，也就是把原始数据投放到 [0,1] 范围内。
            新数值 =（原数值 - 极小值）/（极大值 - 极小值）。
"""
# 初始化数据，每一行表示一个样本，每一列表示一个特征
x = np.array([[ 0., -3.,  1.],
              [ 3.,  1.,  2.],
              [ 0.,  1., -1.]])
# 将数据进行[0,1]规范化
min_max_scaler = preprocessing.MinMaxScaler()
minmax_x = min_max_scaler.fit_transform(x)
"""
[[0.         0.         0.66666667]
 [1.         1.         1.        ]
 [0.         1.         0.        ]]
"""
print(minmax_x)
print("-----------------------")


"""
Z-Score 规范化在 SciKit-Learn 库中使用 preprocessing.scale() 函数，可以直接将给定数据进行 Z-Score 规范化。
Z-Score 规范化可以直接将数据转化为正态分布的情况
        新数值 =（原数值 - 均值）/ 标准差。
"""

# 初始化数据
x = np.array([[ 0., -3.,  1.],
              [ 3.,  1.,  2.],
              [ 0.,  1., -1.]])
# 将数据进行Z-Score规范化
scaled_x = preprocessing.scale(x)
"""
[[-0.70710678 -1.41421356  0.26726124]
 [ 1.41421356  0.70710678  1.06904497]
 [-0.70710678  0.70710678 -1.33630621]]
"""
# 这个结果实际上就是将每行每列的值减去了平均值，再除以方差的结果
print (scaled_x)
print("---------------")
"""
小数定标规范化
小数定标规范化我们需要用 NumPy 库来计算小数点的位数
"""
# 初始化数据
x = np.array([[ 0., -3.,  1.],
              [ 3.,  1.,  2.],
              [ 0.,  1., -1.]])
# 小数定标规范化
j = np.ceil(np.log10(np.max(abs(x))))
scaled_x = x/(10**j)
"""
[[ 0.  -0.3  0.1]
 [ 0.3  0.1  0.2]
 [ 0.   0.1 -0.1]]
"""
print(scaled_x)
