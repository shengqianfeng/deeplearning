#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : partial_derivative_comp.py
@Author : jeffsheng
@Date : 2020/2/14 0014
@Desc :使用python来计算多元函数的偏导数
"""

def f(x,y):
    """
    原函数
    :param x:
    :param y:
    :return:
    """
    return x**2-y**2

def grad_x(f, x, y):
    """
    对x的偏导数
    :param f:
    :param x:
    :param y:
    :return:
    """
    h = 1e-4
    return (f(x + h/2, y) - f(x - h/2, y)) / h


def grad_y(f, x, y):
    """
    对y的偏导数
    :param f:
    :param x:
    :param y:
    :return:
    """
    h = 1e-4
    return (f(x, y + h/2) - f(x, y - h/2)) / h


print(grad_x(f, -1, -1))
print(grad_y(f, -1, -1))




