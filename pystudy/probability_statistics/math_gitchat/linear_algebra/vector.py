#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : vector.py
@Author : jeffsheng
@Date : 2020/2/29 0029
@Desc : 
"""

import numpy as np

A = np.array([1,2,3,4])
print(A)
print(A.shape) # (4,)
print("---------------")
A = np.array([1,2,3,4])
print(A.transpose())
print(A.shape) # (4,)
print("---------------")

A = np.array([[1, 2, 3]])
# [[1 2 3]]
print(A)
print(A.shape)  # (1, 3)
"""
[[1]
 [2]
 [3]]
"""
print(A.T)
print(A.T.shape)    # (3, 1)

print("------------向量的加法-------------")
u = np.array([[1,2,3]]).T
v = np.array([[5,6,7]]).T
"""
[[ 6]
 [ 8]
 [10]]
"""
print(u + v)
print("------------向量的数量乘法-------------")
u = np.array([[1, 2, 3]]).T
"""
[[3]
 [6]
 [9]]
"""
print(3*u)
print("--------向量的内积(点乘)----------")
u = np.array([3, 5, 2])
v = np.array([1, 4, 7])
# 37
# python 内积运算函数 dot 中的参数要求必须是一维行向量，否则报错，因为我们的表示方法用的二维数组表示的向量其实还是矩阵，那么dot函数其实就是矩阵运算
print(np.dot(u, v))

print("--------------矩阵乘法运算-------")
u = np.array([[3, 5, 2]])
v = np.array([[1, 4, 7]]).T
print(np.dot(u,v))

print("------------向量的外积（叉乘）--------")
u = np.array([3, 5])
v = np.array([1, 4])
# 7
print(np.cross(u, v))

# 三维空间中，外积的计算要相对复杂一些，其计算的结果是一个向量而不是一个数值
x = np.array([3, 3, 9])
y = np.array([1, 4, 12])
# [  0 -27 9]
print(np.cross(x, y))


print("-----------向量的线性组合-------------")
u = np.array([[1, 2, 3]]).T
v = np.array([[4, 5, 6]]).T
w = np.array([[7, 8, 9]]).T
"""
[[54]
 [66]
 [78]]
"""
print(3*u+4*v+5*w)


