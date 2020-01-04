# coding: utf-8
# cf.http://d.hatena.ne.jp/white_wheels/20100327/p3
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

"""
求二维数组2*324数组每一行数组每一个元素的梯度
"""
def _numerical_gradient_no_batch(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    # 每次只修改数组中下标idx的那个元素值求此对元素的梯度
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        
    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)# 2*324
        for idx, x in enumerate(X):# 求2*324每一行数组列表中每个元素的梯度
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad# 2*324


def function_2(x):
    if x.ndim == 1:# 1*324
        return np.sum(x**2)# 求一维矩阵所有元素的平方和
    else:
        return np.sum(x**2, axis=1)


def tangent_line(f, x):
    d = numerical_gradient(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y


if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)   # 创建等差数组
    x1 = np.arange(-2, 2.5, 0.25)   #  创建等差数组
    X, Y = np.meshgrid(x0, x1)
    
    X = X.flatten() # 1*324
    Y = Y.flatten() # 1*324
    XY = np.array([X, Y]) # 2*324
    grad = numerical_gradient(function_2, XY)
    
    plt.figure()
    # X，Y代表箭头的位置，-grad[0]（箭头矢量的x分量）, -grad[1]（箭头矢量的y分量）代表箭头数据
    plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")#,headwidth=10,scale=40,color="#444444")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()
    # 重要性质：梯度指示的方向是各点处的函数值减小最多的方向



