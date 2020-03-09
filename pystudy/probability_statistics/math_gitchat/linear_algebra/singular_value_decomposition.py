#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : singular_value_decomposition.py
@Author : jeffsheng
@Date : 2020/3/9 0009
@Desc : 通过 Python 提供的工具，直接一次性获得奇异值分解的所有结果
"""
import numpy as np
A=[[0, 0, 0, 2, 2],
   [0, 0, 0, 3, 3],
   [0, 0, 0, 1, 1],
   [1, 1, 1, 0, 0],
   [2, 2, 2, 0, 0],
   [5, 5, 5, 0, 0],
   [1, 1, 1, 0, 0]]

U, sigma, VT = np.linalg.svd(A)

print(U)
print("-------------")
# sigma是奇异值按照从大到小顺序组成的列表
print(sigma)
print("-----打印V的转置--------")
print(VT)

print("-----------------------")
# 利用奇异值分解的结果进行行压缩和列压缩
A=[[0, 0, 0, 2, 2],
   [0, 0, 0, 3, 3],
   [0, 0, 0, 1, 1],
   [1, 1, 1, 0, 0],
   [2, 2, 2, 0, 0],
   [5, 5, 5, 0, 0],
   [1, 1, 1, 0, 0]]

U, sigma, VT = np.linalg.svd(A)

U_T_2x7 = U.T[:2,:]
print(np.dot(U_T_2x7,A))

VT_2x5=VT[:2,:]
print(np.dot(VT_2x5,np.mat(A).T).T)


print("-----------------------------")
# 利用数据压缩进行矩阵近似
A=[[0, 0, 0, 2, 2],
   [0, 0, 0, 3, 3],
   [0, 0, 0, 1, 1],
   [1, 1, 1, 0, 0],
   [2, 2, 2, 0, 0],
   [5, 5, 5, 0, 0],
   [1, 1, 1, 0, 0]]

U, sigma, VT = np.linalg.svd(A)
A_1 = sigma[0]*np.dot(np.mat(U[:, 0]).T, np.mat(VT[0, :]))
A_2 = sigma[1]*np.dot(np.mat(U[:, 1]).T, np.mat(VT[1, :]))
print(A_1+A_2)