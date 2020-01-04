#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : pie_chart.py
@Author : jeffsheng
@Date : 2019/11/7
@Desc : 饼图
"""


import matplotlib.pyplot as plt
# 数据准备
nums = [25, 37, 33, 37, 6]
labels = ['High-school','Bachelor','Master','Ph.d', 'Others']
# 用Matplotlib画饼图
plt.pie(x = nums, labels=labels)
plt.show()

