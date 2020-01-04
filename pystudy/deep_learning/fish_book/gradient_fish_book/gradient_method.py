# coding: utf-8
import numpy as np
import matplotlib.pylab as plt
"""
测试：经过20次梯度下降后求得的[x0,x1]下降的轨迹图像
样本：np.array([-3.0, 4.0])   
"""

def numerical_gradient(f, x):
    # 0.0001
    h = 1e-4
    # 生成一个形状和x相同、所有元素都为0的数组
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)的计算
        x[idx] = tmp_val + h
        fxh1 = f(x)
        # f(x-h)的计算
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 还原值
    return grad
"""
学习率：
①像学习率这样的参数称为超参数
②相对于神经网络的权重参数是通过训练数据和学习算法自动获得的
③学习率这样的超参数则是人工设定的。一般来说，超参数需要尝试多个值，以便找到一种可以使学习顺利进行的设定
"""
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []  # 记录每次梯度下降后的数值
    for i in range(step_num):
        x_history.append( x.copy() )
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x, np.array(x_history)# x为最终梯度下降后x的最小值，即最优解

# 函数式
def function_2(x):
    return x[0]**2 + x[1]**2

# 初始值
init_x = np.array([-3.0, 4.0])    
# 学习率
lr = 0.1

# 梯度法的重复次数20次
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)
# 学习率过大 lr=10.0  会发散成一个很大的值，下降的太快了，值变得很大
# x, x_history = gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100)
# 学习率过小lr=1e-10 基本上没怎么更新就结束了 下降的超级慢
# x, x_history = gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100)
print(x)    # [-0.03458765  0.04611686]
# --破折线  b表示蓝色 g绿色
plt.plot([-5, 5], [0,0], '--g')
plt.plot([0,0], [-5, 5], '--r')
# o 实心圈标记
plt.plot(x_history[:,0], x_history[:,1], 'o')
plt.grid()

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()


