#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : heatmap.py
@Author : jeffsheng
@Date : 2019/11/7
@Desc : 热力图
"""


import matplotlib.pyplot as plt
import seaborn as sns
# 数据准备
flights = sns.load_dataset("flights")
data=flights.pivot('year','month','passengers')
# 用Seaborn画热力图
sns.heatmap(data)
plt.show()


