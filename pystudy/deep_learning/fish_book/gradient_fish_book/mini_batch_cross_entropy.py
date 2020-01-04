
import numpy as np

# mini-batch版交叉熵误差
def cross_entropy_error(y, t):
    if y.ndim == 1:# 输出y为单行数据
        t = t.reshape(1, t.size)    # 一行t.size列
        y = y.reshape(1, y.size)    # 一行y.size列

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

# 3*10
t = [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]
# 3*10
y = [[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0],
    [0.1, 0.05, 0.0, 0.6, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0],
    [0.1, 0.05, 0.0, 0.0, 0.65, 0.1, 0.0, 0.1, 0.0, 0.0]]

c = cross_entropy_error(np.array(y), np.array(t))
print(c)    # 0.48414455881499613
"""
当监督数据t是标签形式（非one-hot表示，而是像“2”“7”这样的标签）时，交叉熵误差实现方式
np.arange(batch_size)会生成一个从0到batch_size-1的数组
比如当 batch_size为5时， np.arange(batch_size)会生成一个NumPy 数组 [0, 1, 2, 3, 4]
因为t中标签是以 [2, 7, 0, 9, 4]的形式存储的，所以 y[np.arange(batch_size),t]
能抽出各个数据的正确解标签对应的神经网络的输出
（在这个例子中，y[np.arange(batch_size), t] 会 生 成 NumPy 数 组 [y[0,2], y[1,7], y[2,0],y[3,9], y[4,4]]）
每一份batchsize数据中抽出相应的神经网络的输出
"""
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

