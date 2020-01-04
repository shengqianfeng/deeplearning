#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : gradient_descent_question.py
@Author : jeffsheng
@Date : 2019/11/29
@Desc : 梯度下降存在的问题
    学习率过大或过小对最终最优解都有影响
"""


import pystudy.nn_study.d2lzh as d2l

eta = 0.4

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)


"""
#学习率小求最优解速度慢#
# 需要一个较小的学习率从而避免自变量在竖直方向上越过目标函数最优解。
# 然而，这会造成自变量在水平方向上朝最优解移动变慢
"""
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))


"""
#学习率过大越过最优解#
"""
# 试着将学习率调得稍大一点，此时自变量在竖直方向不断越过最优解并逐渐发散
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
