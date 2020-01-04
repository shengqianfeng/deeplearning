#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : bivariate_distribution.py
@Author : jeffsheng
@Date : 2019/11/7
@Desc : 二元变量分布图
"""


import matplotlib.pyplot as plt
import seaborn as sns
# 数据准备
tips = sns.load_dataset("tips")
print(tips.head(10))
# 用Seaborn画二元变量分布图（散点图，核密度图，Hexbin图）
sns.jointplot(x="total_bill", y="tip", data=tips, kind='scatter')
sns.jointplot(x="total_bill", y="tip", data=tips, kind='kde')
sns.jointplot(x="total_bill", y="tip", data=tips, kind='hex')
plt.show()


