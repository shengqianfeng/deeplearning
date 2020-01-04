#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pystudy.nn_study.d2lzh as d2l
import numpy as np

"""
@File : gradient_descent.py
@Author : jeffsheng
@Date : 2019/11/29
@Desc : 《动手学深度学习》梯度下降法演示，观察学习率对自变量下降速度的影响
    
    函数：f(x)  = x * x
    导数：f'(x) = 2 * x
"""
# 使用 x=10 作为初始值，并设学习率： η=0.2
def gd(eta):
    x = 10
    results = [x]
    # 使用梯度下降对 x 迭代10次，可见最终 x 的值较接近最优解
    for i in range(10):
        x -= eta * 2 * x
        results.append(x)
    print('epoch 10, x:', x)
    return results


res = gd(0.2)
print(res)

# 绘制出自变量 x 的迭代轨迹
def show_trace(res):
    # 求出自变量列表中绝对值的最大值，如果小于10则取10为范围数
    n = max(abs(min(res)), abs(max(res)), 10)
    f_line = np.arange(-n, n, 0.1)
    # 设置图像尺寸
    d2l.set_figsize()
    # 还原函数图像y=x*x，定义域：[-10,10]
    d2l.plt.plot(f_line, [x * x for x in f_line])
    # 绘制自变量sgd下降轨迹为定义域的函数图像 并使用圆点连接
    d2l.plt.plot(res, [x * x for x in res], '-o')
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('f(x)')
    d2l.plt.show()
# 学习率为0.2
# show_trace(res)


# 学习率0.05
# 观察减小学习率之后的梯度下降轨迹图  可见学习率过小导致迭代10次后没有到达最小值
show_trace(gd(0.05))

# 学习率1.1
# 观察增大学习率之后的梯度下降轨迹  可见学习率过大导致迭代10次中不断越过最优解x=0逐渐发散
show_trace(gd(1.1))


