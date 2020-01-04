"""

X的每一行都一样，Y的每一列都一样。基于这种强烈的规律性，
numpy提供的numpy.meshgrid()函数可以让我们快速生成坐标矩阵X，Y
就比如：
x = np.array([[0, 1, 2, 3],
              [0, 1, 2, 3],
              [0, 1, 2, 3],
              [0, 1, 2, 3]])
y = np.array([[0, 0, 0, 0],
              [1, 1, 1, 1],
              [2, 2, 2, 2],
              [3, 3, 3, 3]])
              
语法：X,Y = numpy.meshgrid(x, y)
输入的x，y，就是网格点的横纵坐标列向量（非矩阵）
输出的X，Y，就是坐标矩阵。
"""
# 使用meshgrid演示meshgrid.py
import numpy as np
import matplotlib.pyplot as plt
#
# x = np.array([0, 1, 2])
# y = np.array([0, 1])
#
# X, Y = np.meshgrid(x, y)
"""
[[0 1 2]
 [0 1 2]]
"""
# print(X)
"""
[[0 0 0]
 [1 1 1]]
"""
# print(Y)
#
#
# plt.plot(X, Y,
#          color='red',  # 全部点设置为红色
#          marker='.',  # 点的形状为圆点
#          linestyle='')  # 线型为空，也即点与点之间不用线连接
# plt.grid(True)
# plt.show()

# 演示等差数组x和y组成的网格图
x = np.linspace(0,1000, 20)
y = np.linspace(0,500, 20)

X,Y = np.meshgrid(x, y)

plt.plot(X, Y,
         color='limegreen',  # 设置颜色为limegreen
         marker='.',  # 设置点类型为圆点
         linestyle='')  # 设置线型为空，也即没有线连接点
plt.grid(True)
plt.show()








