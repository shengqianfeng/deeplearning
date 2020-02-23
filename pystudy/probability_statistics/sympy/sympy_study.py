#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : sympy_study.py
@Author : jeffsheng
@Date : 2020/2/15 0015
@Desc : SymPy 库是 Python 的数学符号计算库，用它可以进行数学表达式的符号推导和计算，
可以很方便地进行公式推导、级数展开、积分、微分以及解方程等重要运算
"""

from sympy import *
# E**(I*pi)+1=0
print("E**(I*pi)+1={}".format(E**(I*pi)+1))

# 利用 Symbol 函数定义了一个实变量符号 x，并基于它定义了一个新的表达式e^ix，然后我们使用 expand 方法在复数的范围内将
# e^ix 展开为了实部 + 虚部的形式：I*sin(x) + cos(x)
x = Symbol('x', real=True)
print(expand(E**(I*x), complex=True))

# 泰勒展开
# 下面分别对 sin(x) 和 cos(x) 进行 10 阶泰勒展开
x = Symbol('x', real=True)
sin_s = series(sin(x), x, 0, 10)
cos_s = series(cos(x), x, 0, 10)
# sin(x)=x - x**3/6 + x**5/120 - x**7/5040 + x**9/362880 + O(x**10)
print('sin(x)={}'.format(sin_s))
# cos(x)=1 - x**2/2 + x**4/24 - x**6/720 + x**8/40320 + O(x**10)
print('cos(x)={}'.format(cos_s))


# 自定义符号替换
x = Symbol('x', real=True)
y = Symbol('y', real=True)
cos_s = series(cos(x), x, 0, 10)
# cos(x)=1 - x**2/2 + x**4/24 - x**6/720 + x**8/40320 + O(x**10)
print('cos(x)={}'.format(cos_s))
cos_s = cos_s.subs(x, y)
# cos(y)=1 - y**2/2 + y**4/24 - y**6/720 + y**8/40320 + O(y**10)
print('cos(y)={}'.format(cos_s))


# 求导与微分
# 利用 SymPy 库里的 diff 函数对函数进行求导运算
x = Symbol('x', real=True)
y = Symbol('y', real=True)
# 2*cos(2*x)
print(diff(sin(2*x),x))
# 2
print(diff(x**2+2*x+1,x,2))
# 4*x*y
print(diff(x**2*y**2+2*x**3+y**2,x,1,y,1))


# 解方程
# SymPy 库当中的 solve 方法可以用来解方程
x = Symbol('x', real=True)

f1 = x + 1
f2 = x**2+3*x+2
# [-1]
print(solve(f1))
# [-2, -1]
print(solve(f2))


# 表达式求值
# 表达式求值的本质也是符号变量的替换，就是把自定义的符号变量 x 替换成目标取值 2
x = Symbol('x', real=True)
f = x**2+3*x+2
# 12
print(f.subs(x,2))

