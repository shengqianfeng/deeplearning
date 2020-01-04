#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : gradient.py
@Author : jeffsheng
@Date : 2019/11/28
@Desc : 介绍如何使用MXNet提供的autograd模块来自动求梯度
"""

from mxnet import autograd, nd

# 创建一个四行一列的二维向量
x = nd.arange(4).reshape((4, 1))
"""
[[0.]
 [1.]
 [2.]
 [3.]]
<NDArray 4x1 @cpu(0)>
"""
print(x)

# 求x的梯度，先调用attach_grad()函数来申请存储梯度所需要的内存
x.attach_grad()

# 为了减少计算和内存开销，默认条件下MXNet不会记录用于求梯度的计算。我们需要调用record函数来要求MXNet记录与求梯度有关的计算。
# 在调用record函数后，MXNet会记录并计算梯度
with autograd.record():
    y = 2 * nd.dot(x.T, x)


# x的形状为（4, 1），y是一个标量。接下来我们可以通过调用backward函数自动求梯度
# 需要注意的是，如果y不是一个标量，MXNet将默认先对y中元素求和得到新的变量，再求该变量有关x的梯度。
y.backward()

# 验证求得梯度是否正确：L2范数取标量是否为0
assert (x.grad - 4 * x).norm().asscalar() == 0
"""
[[ 0.]
 [ 4.]
 [ 8.]
 [12.]]
<NDArray 4x1 @cpu(0)>

"""
print(x.grad)


print("=====================================")
"""
训练模式和预测模式的自动切换  
"""
# 在调用autograd.record()函数后，MXNet会记录并计算梯度。
# 默认情况下autograd还会将运行模式从预测模式转为训练模式。
# 这可以通过调用is_training函数来查看。
print(autograd.is_training())   #  false
with autograd.record():
    print(autograd.is_training()) # true



