#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : feature_val_vector.py
@Author : jeffsheng
@Date : 2020/3/9
@Desc : 利用python求解特征值和特征向量

"""
import numpy as np
from scipy import linalg

A = np.array([[2, 1],
              [1, 2]])

evalue, evector = linalg.eig(A)
# 特征值3和1
print(evalue)
"""
Python 程序里结果数字不太好看，实质上是因为被处理成模长为 1 的单位向量了。
[[ 0.70710678 -0.70710678]
 [ 0.70710678  0.70710678]]
"""
print(evector)





