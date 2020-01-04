#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : and_gate.py
@Author : jeffsheng
@Date : 2019/11/19
@Desc :
与门：and      仅在x1和x2均为1的时候输出1，其他时候输出0
与非门：not and仅当x1和x2同时为1输出0，其他时候输出1
或门：or 只要一个输入信号为1，输出就为1
"""

#定义与门
def AND(x1,x2):
    w1,w2,theta = 0.5,0.5,0.7
    tmp = x1*w1+x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


import numpy as np
# 使用权重和偏置的计算方式定义与门
def AND2(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
   #b 偏置是调整神经元被激活的容易程度
    b = -0.7
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    else:
        return 1

# 实现与非门
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5]) # 仅权重和偏置与AND不同！
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
     return 1

# 实现或门
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])  # 仅权重和偏置与AND不同！
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1



"""
感知机的局限性：
    感知机的局限性就在于它只能表示由一条直线分割的空间。
异或门：仅当x1或x2中的一方为1时，才会输出1（“异或”是拒绝其他的意思）
异或门无法用一条直接来分割空间（线性空间），而只能用曲线来分割空间（非线性空间）

也就是，感知机无法表示异或门,但是可以通过叠加层来解决，叠加了多层的感知机叫多层感知机
"""
def XOR(x1,x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

print(XOR(0,0)) #0
print(XOR(0,1)) #1
print(XOR(1,0)) #1
print(XOR(1,1)) #0

#结论： 感知机通过叠加层能够进行非线性的表示，理论上还可以表示计算机进行的处理

# 研究证明，
# 2层感知机（严格地说是激活函数使用了非线性的sigmoid函数的感知机，具
# 体请参照下一章）可以表示任意函数。

"""
神经网络的出现就是为了解决使用感知机表示复杂函数时权重的人工设定问题，其
重要性质就是可以自动从数据中学习到合适的权重参数。
"""








