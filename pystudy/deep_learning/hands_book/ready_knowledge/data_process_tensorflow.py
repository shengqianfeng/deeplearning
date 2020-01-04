#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : data_process_tensorflow.py
@Author : jeffsheng
@Date : 2020/1/1
@Desc : tensorflow处理数据
"""
import tensorflow as tf

# 2.0.0
print(tf.__version__)


# 使用tensorflow创建一个行向量
x = tf.constant(range(12))
# (12,)
print(x.shape)
# 12
print(len(x))

X = tf.reshape(x,(3,4))
"""
tf.Tensor(
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]], shape=(3, 4), dtype=int32)
"""
print(X)
"""
tf.Tensor(
[[[0. 0. 0. 0.]
  [0. 0. 0. 0.]
  [0. 0. 0. 0.]]

 [[0. 0. 0. 0.]
  [0. 0. 0. 0.]
  [0. 0. 0. 0.]]], shape=(2, 3, 4), dtype=float32)
"""
print(tf.zeros((2,3,4)))

"""
tf.Tensor(
[[1. 1. 1. 1.]
 [1. 1. 1. 1.]
 [1. 1. 1. 1.]], shape=(3, 4), dtype=float32)
"""
print(tf.ones((3,4)))

"""
[[2 1 4 3]
 [1 2 3 4]
 [4 3 2 1]], shape=(3, 4), dtype=int32)
"""
Y = tf.constant([[2,1,4,3],[1,2,3,4],[4,3,2,1]])
print(Y)

print(tf.random.normal(shape=[3,4], mean=0, stddev=1))


print("--------tensorflow矩阵运算-------------")
"""
# 按元素加法
tf.Tensor(
[[ 2  2  6  6]
 [ 5  7  9 11]
 [12 12 12 12]], shape=(3, 4), dtype=int32)
"""
print(X + Y)

# 按元素乘法
"""
tf.Tensor(
[[ 0  1  8  9]
 [ 4 10 18 28]
 [32 27 20 11]], shape=(3, 4), dtype=int32)
"""
print(X * Y)

# 按元素除法
"""
tf.Tensor(
[[ 0.    1.    0.5   1.  ]
 [ 4.    2.5   2.    1.75]
 [ 2.    3.    5.   11.  ]], shape=(3, 4), dtype=float64)
"""
print(X / Y)

# 按元素做指数运算
"""
tf.Tensor(
[[ 7.389056   2.7182817 54.598152  20.085537 ]
 [ 2.7182817  7.389056  20.085537  54.598152 ]
 [54.598152  20.085537   7.389056   2.7182817]], shape=(3, 4), dtype=float32)
"""
Y = tf.cast(Y, tf.float32)
print(tf.exp(Y))

# 矩阵乘法
Y = tf.cast(Y, tf.int32)
"""
tf.Tensor(
[[ 18  20  10]
 [ 58  60  50]
 [ 98 100  90]], shape=(3, 3), dtype=int32)
"""
print(tf.matmul(X, tf.transpose(Y)))


# 矩阵连接
"""
tf.Tensor(
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [ 2  1  4  3]
 [ 1  2  3  4]
 [ 4  3  2  1]], shape=(6, 4), dtype=int32) 
 
 tf.Tensor(
[[ 0  1  2  3  2  1  4  3]
 [ 4  5  6  7  1  2  3  4]
 [ 8  9 10 11  4  3  2  1]], shape=(3, 8), dtype=int32)
"""
print(tf.concat([X,Y],axis = 0), tf.concat([X,Y],axis = 1))

# 条件判断
"""
tf.Tensor(
[[False  True False  True]
 [False False False False]
 [False False False False]], shape=(3, 4), dtype=bool)
"""
print(tf.equal(X,Y))

# 对tensor中所有元素求和得到只有一个元素的tensor
# tf.Tensor(66, shape=(), dtype=int32)
print(tf.reduce_sum(X))

# 求X的L2范数
X = tf.cast(X, tf.float32)
# tf.Tensor(22.494444, shape=(), dtype=float32)
print(tf.norm(X))


print("-----------广播机制------------")
# 定义并打印两个张量
A = tf.reshape(tf.constant(range(3)), (3,1))
B = tf.reshape(tf.constant(range(2)), (1,2))
"""
tf.Tensor(
[[0]
 [1]
 [2]], shape=(3, 1), dtype=int32) 
 
tf.Tensor([[0 1]], shape=(1, 2), dtype=int32)
"""
print(A,B)

"""
广播原理：
    由于A和B分别是3行1列和1行2列的矩阵，如果要计算A + B，那么A中第一列的3个元素被广播（复制）到了第二列，
而B中第一行的2个元素被广播（复制）到了第二行和第三行。如此，就可以对2个3行2列的矩阵按元素相加
tf.Tensor(
[[0 1]
 [1 2]
 [2 3]], shape=(3, 2), dtype=int32)

"""
print(A + B)

print("----------索引---------")
"""
tf.Tensor(
[[ 4.  5.  6.  7.]
 [ 8.  9. 10. 11.]], shape=(2, 4), dtype=float32)
"""
print(X[1:3])

"""
指定tensor中需要访问的单个元素的位置，如矩阵中行和列的索引，并为该元素重新赋值
"""
X = tf.Variable(X)
X[1,2].assign(9)
"""
<tf.Variable 'Variable:0' shape=(3, 4) dtype=float32, numpy=
array([[ 0.,  1.,  2.,  3.],
       [ 4.,  5.,  9.,  7.],
       [ 8.,  9., 10., 11.]], dtype=float32)>

"""
print(X)

"""
我们也可以截取一部分元素，并为它们重新赋值。在下面的例子中，我们为行索引为1的每一列元素重新赋值
"""
X = tf.Variable(X)
X[1:2,:].assign(tf.ones(X[1:2,:].shape, dtype = tf.float32) * 12)
"""
<tf.Variable 'Variable:0' shape=(3, 4) dtype=float32, numpy=
array([[ 0.,  1.,  2.,  3.],
       [12., 12., 12., 12.],
       [ 8.,  9., 10., 11.]], dtype=float32)>
"""
print(X)



print("----------运算的内存开销--------")
X = tf.Variable(X)
Y = tf.cast(Y, dtype=tf.float32)
before = id(Y)

Y = Y + X
print(id(Y) == before)  # False

# 指定内容到特定内存
Z = tf.Variable(tf.zeros_like(Y))
before = id(Z)
# 其实还是为X + Y开了临时内存来存储计算结果，再复制到Z对应的内存
Z[:].assign(X + Y)
print(id(Z) == before)  # True

# 避免临时内存的开销
Z = tf.add(X, Y)
print(id(Z) == before)  # False ？？？？这里是否讲错了，待研究

# 复用内存
before = id(X)
X.assign_add(Y) # X内存复用了
print(id(X) == before)  # True


print("------------tensor和numpy的互换----------------")
import numpy as np

# numpy--->tensor
P = np.ones((2,3))
D = tf.constant(P)
"""
tf.Tensor(
[[1. 1. 1.]
 [1. 1. 1.]], shape=(2, 3), dtype=float64)
"""
print(D)


# tenfor---->numpy
"""
[[1. 1. 1.]
 [1. 1. 1.]]
"""
print(np.array(D))