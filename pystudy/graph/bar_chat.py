#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : bar_chat.py
@Author : jeffsheng
@Date : 2019/11/7
@Desc : 条形图
"""



import matplotlib.pyplot as plt
import seaborn as sns
# 数据准备
x = ['Cat1', 'Cat2', 'Cat3', 'Cat4', 'Cat5']
y = [5, 4, 8, 12, 7]
# 用Matplotlib画条形图
plt.bar(x, y)
plt.show()
# 用Seaborn画条形图
sns.barplot(x, y)
plt.show()

