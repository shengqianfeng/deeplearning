#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : search_doc.py
@Author : jeffsheng
@Date : 2019/11/28
@Desc : 查阅文档
"""


# 打印nd.random模块中所有的成员或属性
# 通常我们可以忽略掉由__开头和结尾的函数（Python的特别对象）或者由_开头的函数（一般为内部函数）
from mxnet import nd
# 通过其余成员的名字我们大致猜测出这个模块提供了各种随机数的生成方法，包括从均匀分布采样（uniform）、从正态分布采样（normal）、从泊松分布采样（poisson）等
print(dir(nd.random))
# ['NDArray', '_Null', '__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '_internal', '_random_helper', 'current_context', 'exponential', 'exponential_like', 'gamma', 'gamma_like', 'generalized_negative_binomial', 'generalized_negative_binomial_like', 'multinomial', 'negative_binomial', 'negative_binomial_like', 'normal', 'normal_like', 'numeric_types', 'poisson', 'poisson_like', 'randint', 'randn', 'shuffle', 'uniform', 'uniform_like']


# 查找特定函数和类的使用
# 想了解某个函数或者类的具体用法时，可以使用help函数。让我们以NDArray中的ones_like函数为例，查阅它的用法
help(nd.ones_like)

