#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pylab as plt
"""
@File : active_fun.py
@Author : jeffsheng
@Date : 2019/11/19
@Desc : 
    激活函数：将输入信号的总和转换为输出信号，这种函数
一般称为激活函数
"""

# 阶跃函数
"""
阶跃函数：以阈值为界，一旦输入超过阈值，就切换输出。
这样的函数称为“阶跃函数”。感知机中使用了阶跃函数作为
激活函数。如果将激活函数从阶跃函数换成其他函数，就可以进入神经网络的世界了
"""
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0


# 可接受数组的阶跃函数
def step_function(x):
    y = x > 0
    return y.astype(np.int)

x= np.array([-1.0,1.0,2.0])
y = x > 0
print(y)    # [False  True  True]

print(y.astype(np.int)) # [0 1 1]



"""
阶跃函数的图形
阶跃函数以0为界，输出从0切换为1（或者从1切换为0）。
它的值呈阶梯式变化，所以称为阶跃函数
step_function()以该NumPy数组为
参数，对数组的各个元素执行阶跃函数运算，并以数组形式返回运算结果
"""
def step_function(x):
    return np.array(x > 0, dtype=np.int)


# 在−5.0到5.0的范围内，以0.1为单位，
# 生成NumPy数组（[-5.0, -4.9, ..., 4.9]）
x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
print(y)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # 指定y轴的范围
plt.show()





