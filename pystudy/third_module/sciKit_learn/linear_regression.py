#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : linear_regression.py
@Author : jeffsheng
@Date : 2019/11/12
@Desc : 代码实现线性回归模型
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# 特征数据 （工作年限）
experiences = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# 目标数据  （薪资水平）
salaries = np.array([103100, 104900, 106800, 108700, 110400, 112300, 114200, 116100, 117800, 119700, 121600])

# 将特征数据集分为训练集和测试集，除了最后 4 个作为测试用例，其他都用于训练
# 训练集
X_train = experiences[:7]
X_train = X_train.reshape(-1, 1)

# 测试集
X_test = experiences[7:]
X_test = X_test.reshape(-1, 1)

# 把目标数据（特征对应的真实值）也分为训练集和测试集
y_train = salaries[:7]
y_test = salaries[7:]

# 创建线性回归模型
regr = linear_model.LinearRegression()

# 用训练集训练模型——看就这么简单，一行搞定训练过程
# 传入训练集的x和y值进行训练
regr.fit(X_train, y_train)

# 用训练得出的模型进行预测
diabetes_y_pred = regr.predict(X_test)

# 将测试结果以图标的方式显示出来
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, diabetes_y_pred, color='blue', linewidth=3)

#设置坐标轴范围
plt.xlim((0, 10))
plt.ylim((0, 150000))
#设置坐标轴名称
plt.xlabel('work year')
plt.ylabel('salary')
#设置坐标轴刻度
my_x_ticks = np.arange(0, 15, 1)
my_y_ticks = np.arange(0, 150000, 10000)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.show()


