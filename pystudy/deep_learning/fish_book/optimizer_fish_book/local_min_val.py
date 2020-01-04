from pystudy.nn_study.d2lzh import *
import numpy as np

"""
局部最小值
对于目标函数f(x)，如果f(x)在x上的值比在x邻近的其他点的值更小，那么f(x)可能是一个局部最小值（local minimum）。
全局最小值
如果f(x)在x上的值是目标函数在整个定义域上的最小值，那么f(x)是全局最小值（global minimum）

f(x)=x⋅cos(πx),−1.0≤x≤2.0,
"""
# 大致找出该函数的局部最小值和全局最小值的位置
def f(x):
    return x * np.cos(np.pi * x)

# 设置图形尺寸
set_figsize((4.5, 2.5))
x = np.arange(-1.0, 2.0, 0.1)
fig, = plt.plot(x, f(x))  # 逗号表示只取返回列表中的第一个元素
"""
 annotate用于在图形上给数据添加文本注解，而且支持带箭头的划线工具，方便我们在合适的位置添加描述信息
Axes.annotate(s, xy, *args, **kwargs)
---s：注释文本的内容
---xy：被注释的坐标点，二维元组形如(x,y)
---xytext：注释文本的坐标点，也是二维元组，默认与xy相同
---arrowprops：箭头的样式，dict（字典）型数据，如果该属性非空，则会在注释文本和被注释点之间画一个箭头。
参考：https://blog.csdn.net/leaf_zizi/article/details/82886755
"""
fig.axes.annotate('local minimum', xy=(-0.3, -0.25), xytext=(-0.77, -1.0), arrowprops=dict(arrowstyle='->'))
fig.axes.annotate('global minimum', xy=(1.1, -0.95), xytext=(0.6, 0.8), arrowprops=dict(arrowstyle='->'))
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
