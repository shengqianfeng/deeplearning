#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : pooling.py
@Author : jeffsheng
@Date : 2019/12/3
@Desc : 池化层的实现
池化层的实现按下面3个阶段进行
1.展开输入数据。
2.求各行的最大值。
3.转换为合适的输出大小。


最大值的计算可以使用 NumPy 的 np.max方法。np.max可以指定
axis参数，并在这个参数指定的各个轴方向上求最大值。比如，如
果写成np.max(x, axis=1)，就可以在输入x的第1维的各个轴方向
上求最大值。

"""
import sys, os
sys.path.append(os.pardir)
from pystudy.common.util import im2col
import numpy as np


class Pooling:
     def __init__(self, pool_h, pool_w, stride=1, pad=0):
         self.pool_h = pool_h
         self.pool_w = pool_w
         self.stride = stride
         self.pad = pad

     def forward(self, x):
         N, C, H, W = x.shape
         out_h = int(1 + (H - self.pool_h) / self.stride)
         out_w = int(1 + (W - self.pool_w) / self.stride)
         # 展开(1)
         col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
         col = col.reshape(-1, self.pool_h*self.pool_w)
         # 最大值(2)
         out = np.max(col, axis=1)
         # 转换(3)
         out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
         return out


