# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from pystudy.dataset.mnist import load_mnist
from pystudy.nn_study.gradient_fish_book.two_layer_net import TwoLayerNet

"""
mini_batch随机梯度下降法的应用
两层神经网络（输入层784---隐藏层50---输出层10）
"""

# 读入数据（60000,784）矩阵，也就是60000个样本，每个样本784个输入元素
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# 构造神经网络 输入层784个神经元   隐藏层50个神经元  输出层10个神经元
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 1000  # 适当设定循环的次数，也就是1000次梯度下降次数
train_size = x_train.shape[0]# 训练数据的样本个数
# mini-batch的大小为100，需要每次从60000个训练数据中随机取出100个数据
batch_size = 100

learning_rate = 0.1# 学习率
train_loss_list = []# 训练样本每次权重的梯度下降后损失列表
train_acc_list = []# 训练样本每次权重的梯度下降后精确率
test_acc_list = []
# 批处理次数epoch大小
iter_per_epoch = max(train_size / batch_size, 1)

# iters_num为梯度更新的次数
for i in range(iters_num):
    print("---------------->i=",i)
    batch_mask = np.random.choice(train_size, batch_size)#60000条数据中找出100个下标位置
    x_batch = x_train[batch_mask]# 找出这100个位置对应的样本数据
    t_batch = t_train[batch_mask]# 找出这100个位置对应的样本标签
    
    # 计算关于权重的梯度 对这个包含100笔数据的mini-batch求梯度，使用随机梯度下降法（SGD）更新参数
    # grad = network.numerical_gradient(x_batch, t_batch)
    # 通过误差反向传播法求梯度 更高效
    grad = network.gradient(x_batch, t_batch)
    
    # 使用学习率更新权重参数（更新次数为：iters_num）
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 每更新一次，都对训练数据计算损失函数的值，并把该值添加到数组中 （添加次数为：iters_num）
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    # 当更新次数为批处理次数epoch的倍数时统计一次数据
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)# 计算总体训练数据的精确度
        test_acc = network.accuracy(x_test, t_test) #计算总体测试数据的精确度
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 绘制图形
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()