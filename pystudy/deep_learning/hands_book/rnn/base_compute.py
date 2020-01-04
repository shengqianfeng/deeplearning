#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : base_compute.py
@Author : jeffsheng
@Date : 2020/1/4 0004
@Desc : rnn神经元的基本计算
"""

import tensorflow as tf
import numpy as np
X, W_xh = tf.random.normal(shape=(3, 1)), tf.random.normal(shape=(1, 4))
H, W_hh = tf.random.normal(shape=(3, 4)), tf.random.normal(shape=(4, 4))
# 以下两者计算方式结果等价
print(tf.matmul(X, W_xh) + tf.matmul(H, W_hh))

print(tf.matmul(tf.concat([X,H],axis=1), tf.concat([W_xh,W_hh],axis=0)))



