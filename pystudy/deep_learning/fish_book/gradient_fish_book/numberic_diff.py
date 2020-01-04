#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : numberic_diff.py
@Author : jeffsheng
@Date : 2019/11/21
@Desc : 导数与偏导数


"""
import numpy as np
import matplotlib.pylab as plt
# 中心差分法求导
def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)


def function_1(x):
    return 0.01*x**2 + 0.1*x


"""
偏导数
"""
# 输入了一个NumPy数组
def function_2(x,y):
    return x ** 2 + y ** 2    # # 或者return np.sum(x**2)



# x = np.arange(0.0, 20.0, 0.1) # 以0.1为单位，从0到20的数组x
# y = function_1(x)
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.plot(x, y)
# plt.show()
# 计算一下这个函数在x = 5和x = 10处的导数
# print(numerical_diff(function_1, 5))    # 0.1999999999990898
# print(numerical_diff(function_1, 10))   # 0.2999999999986347



"""
三维函数的图像f(x,y) = x ** 2 + y ** 2
"""
x = np.array([-3,-2,-1,0,1,2,3])
y = np.array([-3,-2,-1,0,1,2,3])
plt.xlabel("x")
plt.ylabel("f(x)")
fig = plt.figure()  #定义新的三维坐标轴
ax = plt.axes(projection='3d')
X, Y = np.meshgrid(x, y)
Z = function_2(X,Y)
ax.plot_surface(X,Y,Z,cmap='seismic')
plt.show()


# 求x0 = 3, x1 = 4时，关于x0的偏导数
def function_tmp1(x0):
    return x0 * x0 + 4.0 ** 2.0
# print(numerical_diff(function_tmp1, 3.0))   # 6.00000000000378

# 求x0 = 3, x1 = 4时，关于x1的偏导数
def function_tmp2(x1):
    return 3.0 ** 2.0 + x1 * x1
# print(numerical_diff(function_tmp2, 4.0))   # 7.999999999999119




