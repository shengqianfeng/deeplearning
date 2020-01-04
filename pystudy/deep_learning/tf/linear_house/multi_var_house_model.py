#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2019/12/27 0027 下午 10:10 
# @Author : jeffsmile 
# @File : multi_var_house_model.py
# @desc :

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d

df1 = pd.read_csv('data1.csv', names=['square', 'bedrooms', 'price'])
df1.head()
# 使用figure创建一个原始图像
fig = plt.figure()
# 创建一个 Axes3D object
ax = plt.axes(projection='3d')
# 设置 3 个坐标轴的名称
ax.set_xlabel('square')
ax.set_ylabel('bedrooms')
ax.set_zlabel('price')
# 绘制 3D 散点图
ax.scatter3D(df1['square'], df1['bedrooms'], df1['price'], c=df1['price'], cmap='Greens')
plt.show()


print("----------------数据归一化------------------")
def normalize_feature(df):
    """
    归一化方式：减去平均值除以标准差
    :param df:
    :return:
    """
    return df.apply(lambda column: (column - column.mean()) / column.std())

df = normalize_feature(df1)
df.head()

ax = plt.axes(projection='3d')
ax.set_xlabel('square')
ax.set_ylabel('bedrooms')
ax.set_zlabel('price')
ax.scatter3D(df['square'], df['bedrooms'], df['price'], c=df['price'], cmap='Reds')
plt.show()

print("---------------数据处理：添加ones列--------------")
import numpy as np
ones = pd.DataFrame({'ones': np.ones(len(df))})# ones是n行1列的数据框，表示x0恒为1
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 47 entries, 0 to 46
Data columns (total 1 columns):
ones    47 non-null float64
dtypes: float64(1)
memory usage: 456.0 bytes
"""
ones.info()

df = pd.concat([ones, df], axis=1)  # 根据列合并数据
df.head()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 47 entries, 0 to 46
Data columns (total 4 columns):
ones        47 non-null float64
square      47 non-null float64
bedrooms    47 non-null float64
price       47 non-null float64
dtypes: float64(4)
memory usage: 1.5 KB
"""
df.info()

