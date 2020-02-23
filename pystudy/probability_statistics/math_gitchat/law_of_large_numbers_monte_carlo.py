#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : central_limit_monte_carlo.py
@Author : jeffsheng
@Date : 2020/2/19 0019
@Desc : 大数定理的应用---利用蒙特卡洛方法近似圆的面积
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.stats import uniform
import seaborn
seaborn.set()

n = 100000
r = 1.0
o_x, o_y = (0., 0.)

uniform_x = uniform(o_x-r,2*r).rvs(n)
uniform_y = uniform(o_y-r,2*r).rvs(n)

# (uniform_x,uniform_y)距离原点的距离
d_array = np.sqrt((uniform_x - o_x) ** 2 + (uniform_y - o_y) ** 2)
# np.where(condition, x, y)
# 满足条件(condition)，输出x，不满足输出y。
# 统计在圆内的点用1表示，sum累加其点数
res = sum(np.where(d_array < r, 1, 0))
# 核心原理：圆的面积/正方形面积近似于（圆内点数/总点数），用于估算pi的取值
pi = (res / n) /(r**2) * (2*r)**2

fig, ax = plt.subplots(1, 1)
ax.plot(uniform_x, uniform_y, 'ro', markersize=0.3)
plt.axis('equal')
circle = Circle(xy=(o_x, o_y), radius=r, alpha=0.5)
ax.add_patch(circle)

print('pi={}'.format(pi))
plt.show()




