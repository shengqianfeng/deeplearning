#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : im2col.py
@Author : jeffsheng
@Date : 2019/12/3
@Desc : CNN中处理的是4维数据，因此卷积运算的实现看上去会很复
杂，但是通过使用下面要介绍的im2col这个技巧，问题就会变得很简单

im2col是一个函数，将输入数据展开以适合滤波器（权重）
im2col这个名称是“image to column”的缩写，翻译过来就是“从
图像到矩阵”的意思

对于输入数据，将应用滤波器的区域（3维方块）横向展开为1列。im2col会
在所有应用滤波器的地方进行这个展开处理


im2col (input_data, filter_h, filter_w, stride=1, pad=0)
• input_data―由（数据量，通道，高，长）的4维数组构成的输入数据
• filter_h―滤波器的高
• filter_w―滤波器的长
• stride―步幅
• pad―填充
"""

import sys, os
sys.path.append(os.pardir)
from pystudy.common.util import im2col
import numpy as np

x1 = np.random.rand(1, 3, 7, 7) # 批大小为1、通道为3的7 × 7的数据
# (输入矩阵，滤波器高，滤波器长，步幅，填充)
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape) # (9, 75)

x2 = np.random.rand(10, 3, 7, 7) # 10个数据  通道为3的7 × 7的数据
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape) # (90, 75)





