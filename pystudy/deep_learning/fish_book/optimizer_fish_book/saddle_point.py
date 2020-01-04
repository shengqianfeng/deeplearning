"""

鞍点造成的梯度接近或变成零
f(x)=x***3
"""
from pystudy.nn_study.d2lzh import *
import numpy as np

x = np.arange(-2.0, 2.0, 0.1)
fig, = plt.plot(x, x**3)
fig.axes.annotate('saddle point', xy=(0, -0.2), xytext=(-0.52, -5.0),arrowprops=dict(arrowstyle='->'))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

"""
另一个二维空间函数例子
f(x,y)=x**2−y**2.
目标函数在x轴方向上是局部最小值，但在y轴方向上是局部最大值
"""
x, y = np.mgrid[-1: 1: 31j, -1: 1: 31j]
z = x**2 - y**2

# “111”表示“1×1网格，第一子图”
# “234”表示“2×3网格，第四子图”。
# 子图：就是在一张figure里面生成多张子图
# ax =  plt.figure().add_subplot(111)
# 返回Axes实例
# 参数一：子图总行数
# 参数二：子图总列数
# 参数三：子图位置
ax = plt.figure().add_subplot(111, projection='3d')
#xyz表示数据组
"""
rstride、cstride分别表示行列之间的跨度 rcount、ccount分别表示行列之间的间隔个数，
两组参数只能二选一设置，如果同时设置，会抛出ValueError异常错误
"""
ax.plot_wireframe(x, y, z, **{'rstride': 2, 'cstride': 2})
ax.plot([0], [0], [0], 'rx')# (0,0,0)红色的点
ticks = [-1,  0, 1]
plt.xticks(ticks)
plt.yticks(ticks)
ax.set_zticks(ticks)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

