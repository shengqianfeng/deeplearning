"""
@date 20200302
@desc 使用求向量在空间上的投影方式来求解无解方程组的近似解----最小二乘法
"""
import numpy as np
from scipy import linalg

A = np.array([[2, 1],
              [1, 2],
              [1, 4]])

b = np.array([[4],
              [3],
              [9]])

A_T_A = np.dot(A.T,A)
x = np.dot(np.dot(linalg.inv(A_T_A),A.T),b)

print(x)