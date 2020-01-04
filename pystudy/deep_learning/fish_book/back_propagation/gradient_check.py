# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from pystudy.dataset.mnist import load_mnist
from pystudy.nn_study.gradient_fish_book.two_layer_net import TwoLayerNet

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
# 误差反向传播法求梯度
grad_backprop = network.gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    # 求各个权重参数中对应元素的差的绝对值,并计算平均值
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))

"""
W1:1.837418402261999e-10
b1:9.412343815313443e-10
W2:6.903670755586955e-08
b2:1.3738524516915264e-07
"""