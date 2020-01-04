#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : convolution.py
@Author : jeffsheng
@Date : 2019/12/3
@Desc : 卷积层的实现
"""
import sys, os
sys.path.append(os.pardir)
from pystudy.common.util import im2col
import numpy as np


class Convolution:
     # 初始化滤波器（权重）、偏置、步幅、填充
     def __init__(self, W, b, stride=1, pad=0):
         self.W = W
         self.b = b
         self.stride = stride
         self.pad = pad


     def forward(self, x):
         FN, C, FH, FW = self.W.shape   # 滤波器的四维形状
         N, C, H, W = x.shape
         out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
         out_w = int(1 + (W + 2*self.pad - FW) / self.stride)
         # 用im2col展开输入数据
         col = im2col(x, FH, FW, self.stride, self.pad)
         # 并用reshape将滤波器展开为2维数组  FN滤波器批数量
         # 通过在reshape时指定为-1，reshape函数会自
         # 动计算-1维度上的元素个数，以使多维数组的元素个数前后一致
         """
         比如，
            (10, 3, 5, 5)形状的数组的元素个数共有750个，指定reshape(10,-1)后，就
            会转换成(10, 75)形状的数组。
         """
         col_W = self.W.reshape(FN, -1).T # 滤波器的展开为二维数组
         # 计算展开后的乘积
         out = np.dot(col, col_W) + self.b
         # transpose会更改多维数组的轴的顺序 通过指定从0开始的索引（编号）序列，就可以更改轴的顺序
         out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
         return out




