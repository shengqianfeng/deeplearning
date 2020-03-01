#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : inverse_matrix.py
@Author : jeffsheng
@Date : 2020/3/1 0001
@Desc : 求矩阵的逆
"""
import numpy as np
from scipy import linalg

A = np.array([[1, 35, 0],
              [0, 2, 3],
              [0, 0, 4]])

A_n = linalg.inv(A)
print(A_n)
# A与A的逆矩阵相乘为单位阵
print(np.dot(A, A_n))