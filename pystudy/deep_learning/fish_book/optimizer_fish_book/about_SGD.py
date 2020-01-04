"""
梯度下降法演示
"""
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

def function_1(x,y):
    return 0.05*x**2 + y**2



# 展示函数的3D图像
# x = np.array([-10,-5,0,5,10])
# y = np.array([-10,-5,0,5,10])
# plt.xlabel("x")
# plt.ylabel("y")
# fig = plt.figure()  #定义新的三维坐标轴
# ax = Axes3D(fig)
# X, Y = np.meshgrid(x, y)
# Z = function_1(X,Y)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.plot_surface(X,Y,Z,cmap='seismic')
# plt.show()

"""
对二元函数使用梯度下降。函数：y = 0.05*x^2 + y^2
"""

def function_x(x):
    return 0.1*x


def function_y(y):
    return 2*y


def _numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    # 初始化梯度矩阵为0
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]
        grad[idx] = f(tmp_val)
        x[idx] = tmp_val  # 还原值

    return grad


"""
二元函数的梯度图像
"""
x = np.arange(-10, 11, 1)
y = np.arange(-5, 6, 1)
X, Y = np.meshgrid(x, y)  # 生成网格点坐标矩阵 X,Y都是(21,11)

# X和Y平铺之后为231个数字组成
X = X.flatten()  # 范围[-10,10]重复11次 ---> -10 -9 -8...8 9 10......8 9 10...
Y = Y.flatten()  # [-5,5]重复21次 --->  -5 -5 -5 ........4 4 4 ....5 5 5...

# 分别对X和Y平铺之后的数据进行梯度求导
grad_x = _numerical_gradient(function_x, X)
grad_y = _numerical_gradient(function_y, Y)

plt.figure()
plt.quiver(X, Y, -grad_x, -grad_y, angles="xy", color="#666666")  # ,headwidth=10,scale=40,color="#444444")
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.draw()
plt.show()

