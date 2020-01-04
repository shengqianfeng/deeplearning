#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : gradient_descent.py
@Author : jeffsheng
@Date : 2019/11/21
@Desc : 求函数对自变量数组梯度的计算
样本：
    np.array([3.0, 4.0, 5.0]))
    np.array([0.0, 2.0]))
    np.array([3.0, 0.0]))
梯度：
    像（∂f/∂x0,∂f/∂x₁）这样的由全部变量的偏导数汇总而成的向量称为梯度（gradient）

"""
import numpy as np
"""
求梯度： 对NumPy数组 x的各个元素求数值微分
"""
def numerical_gradient(f, x):
    # 0.0001
    h = 1e-4
    # 生成一个形状和x相同、所有元素都为0的数组
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)的计算
        x[idx] = tmp_val + h
        fxh1 = f(x)
        # f(x-h)的计算
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 还原值
    return grad

# 参数x为数组，其中变化的是其中一个自变量，数组元素
def function_2(x):
    return x[0] ** 2 + x[1] ** 2    # # 或者return np.sum(x**2)

# 求点(3, 4)、 (0, 2)、 (3, 0)处的梯度
print(numerical_gradient(function_2, np.array([3.0, 4.0, 5.0]))) # [6. 8. 0]
print(numerical_gradient(function_2, np.array([0.0, 2.0]))) # [0. 4.]
print(numerical_gradient(function_2, np.array([3.0, 0.0]))) # [6. 0.]

