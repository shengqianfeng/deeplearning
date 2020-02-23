#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : linear_regression.py
@Author : jeffsheng
@Date : 2020/2/22 0022
@Desc : 线性回归方式预测成绩：X = [“Last Score”, “Hours Spent”]，y = “Score”
第15课：逻辑回归——用来做分类的回归模型
"""
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import pandas as pd

# Importing dataset
data = pd.read_csv('quiz.csv', delimiter=',')
used_features = ["Last Score", "Hours Spent"]
X = data[used_features].values
scores = data["Score"].values

X_train = X[:11]
X_test = X[11:]

# Linear Regression - Regression
y_train = scores[:11]
y_test = scores[11:]

regr = LinearRegression()
regr.fit(X_train, y_train)
y_predict = regr.predict(X_test)

print(y_predict)

