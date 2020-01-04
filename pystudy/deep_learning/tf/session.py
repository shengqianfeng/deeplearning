#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : session.py
@Author : jeffsheng
@Date : 2019/12/27
@Desc : tensorflow的session会话
"""
import tensorflow as tf
# 创建数据流图：y = W * x + b，其中W和b为存储节点，x为数据节点。
x = tf.compat.v1.placeholder(tf.float32)
W = tf.compat.v1.Variable(1.0)
b = tf.compat.v1.Variable(1.0)
y = W * x + b

with tf.compat.v1.Session() as sess:
    tf.compat.v1.global_variables_initializer().run() # Operation.run
    fetch = y.eval(feed_dict={x: 3.0})      # Tensor.eval
    print(fetch)                            # fetch = 1.0 * 3.0 + 1.0



