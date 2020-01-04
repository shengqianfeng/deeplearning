#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : box_diagram.py
@Author : jeffsheng
@Date : 2019/11/7
@Desc : 箱线图
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 数据准备
# 生成0-1之间的10*4维度数据
data= np.random.normal(size=(10,4))
lables = ['A','B','C','D']
# 用Matplotlib画箱线图
plt.boxplot(data,labels=lables)
plt.show()
# 用Seaborn画箱线图
df = pd.DataFrame(data, columns=lables)
sns.boxplot(data=df)
plt.show()


