#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2019/12/27 0027 下午 9:45 
# @Author : jeffsmile 
# @File : one_var_house_model.py
# @desc :单变量房价预测模型

"""
warn:
seaborn是基于matplotlib进行的更上一层的封装，需要借助matplotlib中的pyplot 进行展示图片
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 设置上下文方式为notebook，风格为白色网格，黑色点
sns.set(context="notebook", style="whitegrid", palette="dark")
df0 = pd.read_csv('data0.csv', names=['square', 'price'])
# fit_reg是否需要拟合出函数曲线，height控制图的大小
sns.lmplot('square', 'price', df0, height=6, fit_reg=True)
plt.show()

"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 47 entries, 0 to 46
Data columns (total 2 columns):
square    47 non-null int64
price     47 non-null int64
dtypes: int64(2)
memory usage: 832.0 bytes
"""
df0.info()

