# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
import numpy as np
from pystudy.common.functions import softmax, cross_entropy_error
from pystudy.common.gradient import numerical_gradient

"""
权重的梯度：
----->>>>(重要观点)
确定了学习率和输入参数X、结果标签t后，W矩阵中每一个w的变化对于损失函数的变化的影响。这就是神经网络的学习！
以下代码加深理解重要观点 by jeffsheng
输入样本：x = np.array([0.6, 0.9])
初始权重：2*3 随机值生成

================================================================================================
神经网络学习4步骤：
====================================mini batch随机梯度下降法============================================================
步骤1（ mini-batch）
    从训练数据中随机选出一部分数据，这部分数据称为mini-batch。我们的目标是减小mini-batch训练样本的损失函数的值。
步骤2（计算梯度）
    为了减小mini-batch训练样本的损失函数的值，需要求出各个权重参数的梯度。
    梯度表示损失函数的值减小最多的方向。
步骤3（更新参数）
    将权重参数沿梯度方向进行微小更新。
步骤4（重复）
    重复步骤1、步骤2、步骤3。
===========================================================================================
[SGD随机梯度下降法]：使用的数据是随机选择的mini batch数据，称为mini batch随机梯度下降法（mini batch stochastic gradient descent），
 即对随机选择的数据进行的梯度下降法
 
 
 mini batch的大小选择：
 1 如果等于样本总大小m，就相当于batch梯度下降法，轨迹比较平滑，每次循环会消耗很多的时间
 2 如果小于样本总大小m，就是mini_batch随机梯度下降法，有噪音轨迹比较曲折，每次循环时间会更快一些。
 3 极端情况下mini_batch等于1,每次循环只选一个样本进行梯度下降处理。
 对于2和3两种情况永远不会收敛，一直在最优解附近波动。
    
"""
class simpleNet:
    def __init__(self):
        # 生成随机的形状为2× 3的权重参数
        self.W = np.random.randn(2,3)

    # 预测三个神经元的输入值
    def predict(self, x):
        return np.dot(x, self.W)

    # 求损失函数值 这里都是在某个权重值的前提下，计算损失函数值，也就是以损失函数作为权重的求导函数，最终计算权重的梯度
    def loss(self, x, t):
        z = self.predict(x)# 获取三个神经元的输入值
        y = softmax(z)  # 通过激活函数计算输入值得出结果值
        loss = cross_entropy_error(y, t)# 通过神经元激活值得出损失函数值

        return loss

# 输入参数
x = np.array([0.6, 0.9])
# 输入参数正确结果标签
t = np.array([0, 0, 1])

net = simpleNet()

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)# 输入和输出都是2*3权重矩阵，只不过输出是2*3权重的梯度矩阵
"""
[[ 0.24446707  0.25128435 -0.49575141]
 [ 0.3667006   0.37692652 -0.74362712]]
"""
print(dW)
