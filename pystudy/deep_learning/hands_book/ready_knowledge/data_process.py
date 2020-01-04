#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : data_process.py
@Author : jeffsheng
@Date : 2020/1/1
@Desc : mxnet的数据操作类ndarray
"""
from mxnet import nd

# 创建一个行向量
x = nd.arange(12)
"""
[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11.]
<NDArray 12 @cpu(0)>
"""
print(x)
# (12,)
print(x.shape)
# 12
print(x.size)

X = x.reshape((3, 4))
"""
改变行向量的形状
[[ 0.  1.  2.  3.]
 [ 4.  5.  6.  7.]
 [ 8.  9. 10. 11.]]
<NDArray 3x4 @cpu(0)>
"""
print(X)

"""
创建一个各元素为0，形状为(2, 3, 4)的张量
向量和矩阵都是特殊的张量
"""
print(nd.zeros((2, 3, 4)))

# 创建各元素为1的张量
"""
[[1. 1. 1. 1.]
 [1. 1. 1. 1.]
 [1. 1. 1. 1.]]
<NDArray 3x4 @cpu(0)>
"""
print(nd.ones((3, 4)))

# Python的列表（list）指定需要创建的NDArray中每个元素的值
Y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
"""
[[2. 1. 4. 3.]
 [1. 2. 3. 4.]
 [4. 3. 2. 1.]]
<NDArray 3x4 @cpu(0)>
"""
print(Y)

"""
随机生成NDArray中每个元素的值
创建一个形状为(3, 4)的NDArray,它的每个元素都随机采样于均值为0、标准差为1的正态分布
"""
print(nd.random.normal(0, 1, shape=(3, 4)))

print("....................ndarray的矩阵运算...........")
"""
[[ 2.  2.  6.  6.]
 [ 5.  7.  9. 11.]
 [12. 12. 12. 12.]]
<NDArray 3x4 @cpu(0)>
"""
# 按元素相加
print(X + Y)
# 按元素相乘
"""
[[ 0.  1.  8.  9.]
 [ 4. 10. 18. 28.]
 [32. 27. 20. 11.]]
<NDArray 3x4 @cpu(0)>
"""
print(X * Y)
# 按元素除法
print(X/Y)
# 按元素指数运算
print(X.exp())

# 矩阵点乘
"""
[[ 18.  20.  10.]
 [ 58.  60.  50.]
 [ 98. 100.  90.]]
<NDArray 3x3 @cpu(0)>
"""
print(nd.dot(X, Y.T))

# 多个ndarray连接 如下分别按行，列连接
"""
[[ 0.  1.  2.  3.]
 [ 4.  5.  6.  7.]
 [ 8.  9. 10. 11.]
 [ 2.  1.  4.  3.]
 [ 1.  2.  3.  4.]
 [ 4.  3.  2.  1.]]
<NDArray 6x4 @cpu(0)> 
[[ 0.  1.  2.  3.  2.  1.  4.  3.]
 [ 4.  5.  6.  7.  1.  2.  3.  4.]
 [ 8.  9. 10. 11.  4.  3.  2.  1.]]
<NDArray 3x8 @cpu(0)>
"""
print(nd.concat(X, Y, dim=0), nd.concat(X, Y, dim=1))

# 矩阵的条件判断  相同位置值相等为1，反之为0
"""
[[0. 1. 0. 1.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
<NDArray 3x4 @cpu(0)>
"""
print(X == Y)


# ndarray所有元素求和,返回ndarray，但是结果只有一个元素
"""
[66.]
<NDArray 1 @cpu(0)>
"""
print(X.sum())

print("--------asscalar-----------")
x = nd.ones((1,), dtype='int32')
"""
[1]
<NDArray 1 @cpu(0)>
"""
print(x)
# 1  asscalar()：含义是获取x的标量值，x必须满足（1，）形状
print(x.asscalar())
# <class 'numpy.int32'>
print(type(x.asscalar()))
"""
X.norm():计算X的L2范数
[22.494442]
<NDArray 1 @cpu(0)>
"""
print(X.norm())

print("--------------ndrray广播机制--------------")
A = nd.arange(3).reshape((3, 1))
B = nd.arange(2).reshape((1, 2))
"""
A:
[[0.]
 [1.]
 [2.]]
<NDArray 3x1 @cpu(0)> 

B:
[[0. 1.]]
<NDArray 1x2 @cpu(0)>
----------------------
以上A和B两个元素形状不同，如何按照元素相加则会触发广播机制，即A第一列元素复制到第二列变为
        [[0. 0.]
         [1. 1.]
         [2. 2.]]
而B第一行的元素复制到第二三两行上变为：
        [[0. 1.]
        [0. 1.]
        [0. 1.]
        ]
这样以来A+B就可以运算了
"""
print(A,B)
"""
[[0. 1.]
 [1. 2.]
 [2. 3.]]
<NDArray 3x2 @cpu(0)>
"""
print(A+B)

print("--------------ndrray索引原理--------------")
# 索引从0开始，截取X第一、二两行元素
"""
[[ 4.  5.  6.  7.]
 [ 8.  9. 10. 11.]]
<NDArray 2x4 @cpu(0)>
"""
print(X[1:3])

# 为X的单个位置元素赋值
X[1,2]=9
"""
[[ 0.  1.  2.  3.]
 [ 4.  5.  9.  7.]
 [ 8.  9. 10. 11.]]
<NDArray 3x4 @cpu(0)>
"""
print(X)

# 为行索引为1的每一列赋值
X[1:2, :] = 12
"""
[[ 0.  1.  2.  3.]
 [12. 12. 12. 12.]
 [ 8.  9. 10. 11.]]
<NDArray 3x4 @cpu(0)>
"""
print(X)

print("----------ndarray的内存开销-----------")
# 原理： 使用Python自带的id函数：如果两个实例的ID一致，那么它们所对应的内存地址相同；反之则不同
# 比较两个ndrray变量的内存开销
before = id(Y)
Y = Y + X
print(id(Y) == before) # False 说明为Y新开了内存

# 指定结果到特定内存
Z = Y.zeros_like()
before = id(Z)
Z[:] = X + Y
print(id(Z) == before) # True 但是仍然为X + Y开了临时内存来存储计算结果，再复制到Z对应的内存

# 避免存储临时计算结果的开销
nd.elemwise_add(X, Y, out=Z)
print(id(Z) == before) # True

# 复用已有变量的内存空间
before = id(X)
X += Y
print(id(X) == before) # True

print("-----------NDArray和NumPy相互变换¶--------------")
import numpy as np

# numpy---> ndarray
P = np.ones((2, 3))
D = nd.array(P)
"""
[[1. 1. 1.]
 [1. 1. 1.]]
<NDArray 2x3 @cpu(0)>
"""
print(D)


# ndarray---->numpy
"""
[[1. 1. 1.]
 [1. 1. 1.]]
 <class 'numpy.ndarray'>
"""
print(type(D.asnumpy()))







