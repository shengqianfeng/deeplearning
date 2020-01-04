import numpy as np
import matplotlib.pyplot as plt

"""
 计算高度的函数
 :param x: 向量
 :param y: 向量
 :return: dim(x)*dim(y)维的矩阵
 """
def function_1(x,y):
    return 0.05*x**2 + y**2


step=0.01
x0 = np.arange(-10, 10, step)
x1 = np.arange(-10, 10, step)
X,Y = np.meshgrid(x0,x1)  # 获得网格坐标矩阵

# 进行等高线绘制 50表示等高线的密度
c = plt.contour(X, Y, function_1(X,Y), 50, colors='k')
plt.xlabel('x')
plt.ylabel('y')
# plt.clabel(c,inline=True,fontsize=10) # 在等高线上添加数字
plt.show()