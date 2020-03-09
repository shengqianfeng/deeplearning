#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : linear_equations.py
@Author : jeffsheng
@Date : 2020/3/1 0001
@Desc : python求解线性方程组

"""
import numpy as np
from scipy import linalg

A = np.array([[1, 2, 3],
              [1, -1, 4],
              [2, 3, -1]])

y = np.array([14, 11, 5])

x = linalg.solve(A, y)
# [1. 2. 3.]
print(x)


