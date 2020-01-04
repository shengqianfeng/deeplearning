#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : np_pad.py
@Author : jeffsheng
@Date : 2019/12/3
@Desc :
pad(array, pad_width, mode, **kwargs)
返回值：数组

参数说明：
array——表示需要填充的数组；

pad_width——表示每个轴（axis）边缘需要填充的数值数目。
参数输入方式为：（(before_1, after_1), … (before_N, after_N)），其中(before_1, after_1)表示第1轴两边缘分别填充before_1个和after_1个数值。
取值为：{sequence, array_like, int}

mode——表示填充的方式（取值：str字符串或用户提供的函数）,总共有11种填充模式；
            ‘constant’——表示连续填充相同的值，每个轴可以分别指定填充值，constant_values=（x, y）时前面用x填充，后面用y填充，缺省值填充0
            ‘edge’——表示用边缘值填充
            ‘linear_ramp’——表示用边缘递减的方式填充
            ‘maximum’——表示最大值填充
            ‘mean’——表示均值填充
            ‘median’——表示中位数填充
            ‘minimum’——表示最小值填充
            ‘reflect’——表示对称填充
            ‘symmetric’——表示对称填充
            ‘wrap’——表示用原数组后面的值填充前面，前面的值填充后面



"""
import numpy as np



print("------------------------对一维数组的填充-----------------------------------")
array = np.array([1, 2, 3])
print("array",array)
# (1,2)表示在一维数组array前面填充1位，最后面填充2位
#  constant_values=(0,8) 表示前面填充0，后面填充8
ndarray=np.pad(array,(1,2),'constant', constant_values=(0,8))
print("ndarray=",ndarray)






print("-------------------------对二维数组的填充--------------------------------")
# 在卷积神经网络中，通常采用constant填充方式!!
A = np.arange(95,99).reshape(2,2)    #原始输入数组
"""
[[95 96]
 [97 98]]
"""
print(A)
"""
1 在数组A的边缘填充constant_values指定的数值
2  (3,2)表示在A的第[0]轴填充（二维数组中，0轴表示行），即在0轴前面填充3个宽度的0，比如数组A中的95,96两个元素前面各填充了3个0；
在后面填充2个0，比如数组A中的97,98两个元素后面各填充了2个0
  （1,4）表示在A的第[1]轴填充（二维数组中，1轴表示列），即在1轴前面填充1个宽度的0，后面填充4个宽度的0

打印如图：  
[[ 0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0]
 [ 0 95 96  0  0  0  0]
 [ 0 97 98  0  0  0  0]
 [ 0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0]]
"""
B = np.pad(A, ((3,2),(1,4)), 'constant', constant_values = (0,0))
print(B)    # (5, 6, 8)



print("-------------------------对三维数组的填充--------------------------------")
"""
(1,1),(2,2),(3,3)
    第一个元组是y轴，第二个元组是z轴，第三个元组是x轴
解释：
y轴方向前后填充1个0
z轴方向前后填充2个0
x轴方向前后填充3个0
"""
C = np.random.rand(3, 2, 2)
print(C)
D = np.pad(C, ((1,1),(2,2),(3,3)), 'constant', constant_values = (0,0))
print(D.shape)


