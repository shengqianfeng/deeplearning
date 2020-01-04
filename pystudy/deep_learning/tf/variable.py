#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : variable.py
@Author : jeffsheng
@Date : 2019/12/26
@Desc : tensorflow的变量
"""

import tensorflow as tf
import os
# 屏蔽不支持avx2指令警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 创建变量
# tf.random_normal 方法返回形状为(1，4)的张量。它的4个元素符合均值为100、标准差为0.35的正态分布。
# 两维的，第一维一个元素，第二维四个元素
W = tf.Variable(initial_value=tf.random.normal(shape=(1, 4), mean=100, stddev=0.35), name="W")
# 一维的一行四列
b = tf.Variable(tf.zeros([4]), name="b")
"""
结果：
[<tf.Variable 'W:0' shape=(1, 4) dtype=float32_ref>, 
 <tf.Variable 'b:0' shape=(4,) dtype=float32_ref>]
"""
print([W, b])


# 初始化变量
# 创建会话（之后小节介绍）
sess = tf.compat.v1.Session()
# 使用 global_variables_initializer 方法初始化全局变量 W 和 b
sess.run(tf.compat.v1.global_variables_initializer())
# 执行操作，获取变量值
"""
结果：
[array([[100.17326, 100.3, 99.91585, 100.51389]], dtype=float32),
 array([0., 0., 0., 0.], dtype=float32)]
"""
print(sess.run([W, b]))

# 执行更新变量 b 的操作 [1. 1. 1. 1.]
print(sess.run(tf.compat.v1.assign_add(b, [1, 1, 1, 1])))



### Saver 使用示例
# 创建Saver
saver = tf.compat.v1.train.Saver({'W': W, 'b': b})
# 存储变量到文件 './summary/test.ckpt-0'
saver.save(sess, './summary/test.ckpt', global_step=0)

# 再次执行更新变量 b 的操作
sess.run(tf.compat.v1.assign_add(b, [1, 1, 1, 1]))
# 获取变量 b 的最新值 [2. 2. 2. 2.]
print(sess.run(b))


# 从文件中恢复变量 b 的值
saver.restore(sess, './summary/test.ckpt-0')
# 查看变量 b 是否恢复成功  [1. 1. 1. 1.]
print(sess.run(b))

