#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : logistic_regression.py
@Author : jeffsheng
@Date : 2020/2/22 0022
@Desc : 逻辑回归预测
预测学生成绩，合格标志为1，不合格为0
"""
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import pandas as pd

# Importing dataset
data = pd.read_csv('quiz.csv', delimiter=',')

used_features = [ "Last Score", "Hours Spent"]
X = data[used_features].values
scores = data["Score"].values

X_train = X[:11]
X_test = X[11:]

# Logistic Regression – Binary Classification
passed = []

for i in range(len(scores)):
    if(scores[i] >= 60):
        passed.append(1)
    else:
        passed.append(0)

y_train = passed[:11]
y_test = passed[11:]

classifier = LogisticRegression(C=1e5)
classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)
print(y_predict)



