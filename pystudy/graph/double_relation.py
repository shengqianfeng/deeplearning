#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : double_relation.py
@Author : jeffsheng
@Date : 2019/11/7
@Desc : 成对关系
"""


import matplotlib.pyplot as plt
import seaborn as sns
# 数据准备
iris = sns.load_dataset('iris')
# 用Seaborn画成对关系
sns.pairplot(iris)
plt.show()


