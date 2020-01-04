#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : scatter_plot.py
@Author : jeffsheng
@Date : 2019/11/7
@Desc : 散点图的绘制
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 数据准备
N = 1000
# 随机的1000个点
x = np.random.randn(N)
y = np.random.randn(N)
# 用Matplotlib画散点图
plt.scatter(x, y,marker='x')
plt.show()
# 用Seaborn画散点图
df = pd.DataFrame({'x': x, 'y': y})
sns.jointplot(x="x", y="y", data=df, kind='scatter');
plt.show()


