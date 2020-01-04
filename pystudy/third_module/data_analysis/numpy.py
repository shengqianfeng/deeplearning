#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : numpy.py
@Author : jeffsheng
@Date : 2019/11/6
@Desc : numpy
    在 NumPy 学习中，你重点要掌握的就是对数组的使用，因为这是 NumPy 和标准 Python 最大的区别。
在 NumPy 中重新对数组进行了定义，同时提供了算术和统计运算，
你也可以使用 NumPy 自带的排序功能，一句话就搞定各种排序算法
"""


import numpy as np

a = np.array([1, 2, 3])
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b[1,1]=10   # 修改第二维第二列元素为10
# 通过函数 shape 属性获得数组的大小
print(a.shape)
print(b.shape)
# 通过 dtype 获得元素的属性
print(a.dtype)  # int32
"""
[
 [ 1  2  3]
 [ 4 10  6]
 [ 7  8  9]
]
"""
print(b)


"""
结构数组
"""
import numpy as np

# 用 dtype 定义的结构类型
persontype = np.dtype({
                    'names':['name', 'age', 'chinese', 'math', 'english'],
                    'formats':['S32','i', 'i', 'i', 'f']})
# 用 array 中指定了结构数组的类型 dtype=persontype
peoples = np.array([("ZhangFei",32,75,100, 90),("GuanYu",24,85,96,88.5),("ZhaoYun",28,85,92,96.5),("HuangZhong",29,65,85,100)],dtype=persontype)

# 每个人的年龄
ages = peoples[:]['age']
# 每个人的语文成绩
chineses = peoples[:]['chinese']
# 每个人的数学成绩
maths = peoples[:]['math']
# 每个人的语文成绩
englishs = peoples[:]['english']

# 年龄的平均值
print(np.mean(ages))
# 语文成绩的平均值
print(np.mean(chineses))
# 数学成绩的平均值
print(np.mean(maths))
# 英文成绩的平均值
print(np.mean(englishs))


"""
NumPy ufunc
    对数组中每个元素进行函数操作
"""
# np.arange 和 np.linspace 起到的作用是一样的，都是创建等差数组
x1 = np.arange(1,11,2)
x2 = np.linspace(1,9,5)
# arange() 类似内置函数 range()，通过指定初始值、终值、步长来创建等差数列的一维数组，默认是不包括终值的。
print(x1)   # [1 3 5 7 9]
# linspace 是 linear space 的缩写，代表线性等分向量的含义。
# linspace() 通过指定初始值、终值、元素个数来创建等差数列的一维数组，默认是包括终值的
print(x2)   # [1. 3. 5. 7. 9.]




"""
NumPy
    进行加、减、乘、除、求 n 次方和取余数
"""

x1 = np.arange(1,11,2)
x2 = np.linspace(1,9,5)
print(x1)   # [1 3 5 7 9]
print(x2)   # [1. 3. 5. 7. 9.]
print(np.add(x1, x2))   # [ 2.  6. 10. 14. 18.]
print(np.subtract(x1, x2))  # [0. 0. 0. 0. 0.]
print(np.multiply(x1, x2))  # [ 1.  9. 25. 49. 81.]
print(np.divide(x1, x2))    # [1. 1. 1. 1. 1.]
# 在 n 次方中，x2 数组中的元素实际上是次方的次数，x1 数组的元素为基数
print(np.power(x1, x2))     # [1.00000000e+00 2.70000000e+01 3.12500000e+03 8.23543000e+05 3.87420489e+08]
print(np.remainder(x1, x2)) # [0. 0. 0. 0. 0.]

print("------------------")
print(np.arange(3))  # 仅输入stop值，此时start默认从0开始 [0 1 2]
print("------------------")
"""
组 / 矩阵中的最大值函数 amax()，最小值函数 amin()
在 NumPy 中，每一个线性的数组称为一个轴（axes）
在 NumPy 数组中，维数称为秩（rank），一维数组的秩为 1，二维数组的秩为 2，秩就是描述轴的数量
"""
a = np.array([[1,2,3], [4,5,6], [7,8,9]])
# 对于一个二维数组 a，amin(a) 指的是数组中全部元素的最小值
print(np.amin(a))   # 1
# amin(a,0) 是延着 axis=0 轴的最小值，axis=0 轴是把元素看成了 [1,4,7], [2,5,8], [3,6,9] 三个元素，所以最小值为 [1,2,3]
print(np.amin(a,0))     # [1 2 3]
# amin(a,1) 是延着 axis=1 轴的最小值，axis=1 轴是把元素看成了 [1,2,3], [4,5,6], [7,8,9] 三个元素
print(np.amin(a,1))
print(np.amax(a))
print(np.amax(a,0))
print(np.amax(a,1))



"""
统计最大值与最小值之差 ptp()
"""
a = np.array([[1,2,3], [4,5,6], [7,8,9]])
print (np.ptp(a))   # np.ptp(a) 可以统计数组中最大值与最小值的差，即 9-1=8
print(np.ptp(a,0))  # ptp(a,0) 统计的是沿着 axis=0 轴的最大值与最小值之差，即 7-1=6
print(np.ptp(a,1))  # ptp(a,1) 统计的是沿着 axis=1 轴的最大值与最小值之差，即 3-1=2

"""
统计数组的百分位数 percentile()
percentile() 代表着第 p 个百分位数，这里 p 的取值范围是 0-100
如果 p=0，那么就是求最小值，如果 p=50 就是求平均值，如果 p=100 就是求最大值
同样你也可以求得在 axis=0 和 axis=1 两个轴上的 p% 的百分位数。
"""
a = np.array([[1,2,3], [4,5,6], [7,8,9]])
print (np.percentile(a, 50))
print (np.percentile(a, 50, axis=0))    # [4. 5. 6.]
print (np.percentile(a, 50, axis=1))    # print (np.percentile(a, 50, axis=1))    #


"""
统计数组中的中位数 median()、平均数 mean()
"""

a = np.array([[1,2,3], [4,5,6], [7,8,9]])
#求中位数
print (np.median(a)) # 5.0
print (np.median(a, axis=0))    # [4. 5. 6.]
print (np.median(a, axis=1))    # [2. 5. 8.]
#求平均数
print (np.mean(a))  # 5.0
print (np.mean(a, axis=0))     # [4. 5. 6.]
print (np.mean(a, axis=1))      # [2. 5. 8.]


"""
统计数组中的加权平均值 average()
average() 函数可以求加权平均，加权平均的意思就是每个元素可以设置个权重
默认情况下每个元素的权重是相同的，所以 np.average(a)=(1+2+3+4)/4=2.5
你也可以指定权重数组 wts=[1,2,3,4]，这样加权平均
 np.average(a,weights=wts)=(1*1+2*2+3*3+4*4)/(1+2+3+4)=3.0。
"""
a = np.array([1,2,3,4])
wts = np.array([1,2,3,4])
print(np.average(a))    # 2.5
print(np.average(a,weights=wts))    # 3.0

"""
统计数组中的标准差 std()、方差 var()
方差的计算是指每个数值与平均值之差的平方求和的平均值，即 mean((x - x.mean())** 2)
标准差是方差的算术平方根。在数学意义上，代表的是一组数据离平均值的分散程度。
"""
a = np.array([1,2,3,4])
print (np.std(a))
print (np.var(a))

"""
NumPy 排序
使用 sort 函数，sort(a, axis=-1, kind=‘quicksort’, order=None)，
默认情况下使用的是快速排序；在 kind 里，可以指定 quicksort、mergesort、heapsort 分别表示快速排序、合并排序、堆排序。
同样 axis 默认是 -1，即沿着数组的最后一个轴进行排序，也可以取不同的 axis 轴，
或者 axis=None 代表采用扁平化的方式作为一个向量进行排序。

另外 order 字段，对于结构化的数组可以指定按照某个字段进行排序。
"""
import numpy as np
a = np.array([[4,3,2],[2,4,1]])
"""
[[2 3 4]
 [1 2 4]]
"""
print(np.sort(a))

print(np.sort(a, axis=None))    # [1 2 2 3 4 4]
"""
[[2 3 1]
 [4 4 2]]
"""
print(np.sort(a, axis=0))
"""
[[2 3 4]
 [1 2 4]]
"""
print(np.sort(a, axis=1))




"""
# zeros 相似的还有ones
用法：zeros(shape, dtype=float, order='C')

返回：返回来一个给定形状和类型的用0填充的数组；

参数：1)shape:形状
    2)dtype:数据类型，可选参数，默认numpy.float64
            --dtype类型：t ,位域,如t4代表4位
                     b,布尔值，true or false
                     i,整数,如i8(64位）
                     u,无符号整数，u8(64位）
                     f,浮点数，f8（64位）
                     c,浮点负数，
                     o,对象，
                     s,a，字符串，s24
                      u,unicode,u24

    2)order:可选参数{'C'，'F'}，默认C行优先；F代表列优先
"""
import  numpy as np

print(np.zeros(5))  # [0. 0. 0. 0. 0.]
print(np.zeros((5,), dtype=np.int)) # [0 0 0 0 0]
"""
一个看两行一列的数组
[[0.]
 [0.]]
"""
print(np.zeros((2, 1)))
s = (2,2)
"""
两行两列的数组
[[0. 0.]
 [0. 0.]]
"""
print(np.zeros(s))


"""
array和asarray都可将结构数据转换为ndarray类型
但是主要区别就是当数据源是ndarray时，array仍会copy出一个副本，占用新的内存，但asarray不会
"""
import numpy as np

# example 1: 结论：可见array和asarray没有区别，都得元数据进行了复制。
data1 = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
arr2 = np.array(data1)
arr3 = np.asarray(data1)
data1[1][1] = 2
print('data1:\n', data1)    #  [[1, 1, 1], [1, 2, 1], [1, 1, 1]]
"""
 [[1 1 1]
 [1 1 1]
 [1 1 1]]
"""
print('arr2:\n', arr2)
"""
 [[1 1 1]
 [1 1 1]
 [1 1 1]]
"""
print('arr3:\n', arr3)


# example 2:
arr1 = np.ones((3, 3))  # ones()返回一个全1的n维数组，跟zeros参数类似
arr2 = np.array(arr1)
arr3 = np.asarray(arr1)
arr1[1] = 2 # 将第二行元素设置为2
"""
 [[1. 1. 1.]
 [2. 2. 2.]
 [1. 1. 1.]]
"""
print('arr1:\n', arr1)
"""
 [[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]
"""
print('arr2:\n', arr2)
"""
asarray不会拷贝arr1，用的还是arr1，返回的还是arr1
 [[1. 1. 1.]
 [2. 2. 2.]
 [1. 1. 1.]]
"""
print('arr3:\n', arr3)


"""
矩阵的乘方
代表概率转移矩阵中从状态 1 到状态 3，两步状态转移的概率值
"""
A = np.array([[0.7, 0.2, 0.1],
              [0.3, 0.5, 0.2],
              [0.2, 0.4, 0.4]])
"""
[[0.57 0.28 0.15]
 [0.4  0.39 0.21]
 [0.34 0.4  0.26]]
"""
# dot()返回的是两个数组的点积
print(np.dot(A, A))

print("----------------------")
"""
矩阵的n次乘方
"""
A = np.array([[0.7, 0.2, 0.1],
              [0.3, 0.5, 0.2],
              [0.2, 0.4, 0.4]])

def get_matrix_pow(matrix, n):
    ret = matrix
    for i in range(n):
        ret = np.dot(ret,A)
    print(ret)
"""
[[0.4879 0.3288 0.1833]
 [0.4554 0.3481 0.1965]
 [0.4422 0.3552 0.2026]]
"""
get_matrix_pow(A,3)
"""
[[0.471945 0.338164 0.189891]
 [0.465628 0.341871 0.192501]
 [0.463018 0.343384 0.193598]]
"""
get_matrix_pow(A,5)
"""
[[0.46814979 0.34038764 0.19146257]
 [0.46804396 0.34044963 0.1915064 ]
 [0.46800013 0.34047531 0.19152456]]
"""
get_matrix_pow(A,10)
"""
[[0.46808512 0.34042552 0.19148935]
 [0.46808509 0.34042554 0.19148937]
 [0.46808508 0.34042555 0.19148937]]
"""
get_matrix_pow(A,20)
"""
[[0.46808511 0.34042553 0.19148936]
 [0.46808511 0.34042553 0.19148936]
 [0.46808511 0.34042553 0.19148936]]
"""
get_matrix_pow(A,100)
"""
n步状态转移
随着乘方次数的增加，逐渐收敛于
[[0.46808511 0.34042553 0.19148936]
 [0.46808511 0.34042553 0.19148936]
 [0.46808511 0.34042553 0.19148936]]
"""
get_matrix_pow(A,200)