
"""
损失函数：均方差函数和交叉熵函数

"""
import numpy as np
"""
均方差函数
         (参数 y和 t是NumPy数组.y表示神经网络输出结果，t表示正确标签结果)
"""
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# 设“2”为正确解
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# 例1： “2”的概率最高的情况（0.6）
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(mean_squared_error(np.array(y), np.array(t))) # 0.09750000000000003

# # 例2： “7”的概率最高的情况（0.6）
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(mean_squared_error(np.array(y), np.array(t))) # 0.5975
"""
均方差结论：
    第一个例子的损失函数的值更小，和监督数据之间的
    误差较小。也就是说，均方误差显示第一个例子的输出结果与监督数据更加吻合
"""


"""
交叉熵误差：
    交叉熵误差的值是由正确解标签所对应的输出结果决定的
    
    1 参数y和t是NumPy数组
    2 函数内部在计算np.log时，加上了一个微小值delta。
    这是因为，当出现np.log(0)时， np.log(0)会变为负无限大的-inf，
    这样以来就会导致后续计算无法进行。作为保护性对策，添加一个微小值可以防止负无限大的发生
"""
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))    # 0.510825457099338
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))    # 2.302584092994546

