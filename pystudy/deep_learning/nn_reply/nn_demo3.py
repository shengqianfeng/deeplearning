# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle
from pystudy.dataset.mnist import load_mnist
from pystudy.common.functions import sigmoid, softmax

"""
神经网络复盘之演进：三层神经网络mnist训练集的批处理
输入层：784个神经元
输出层：10个神经元
隐藏层1:50个神经元
隐藏层2:100个神经元

演示：读入事先训练好的参数加载模型，预测测试集并计算准确率
"""

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("../dataset/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)
    return y


x, t = get_data()
network = init_network()
batch_size = 100 # 批数量
accuracy_cnt = 0
# 每次从测试集的10000张中取出100张图进行预测。预测结果为(100,10),求出每行最大值得到(1,100)最终结果
for i in range(0, len(x), batch_size):
    x_batch = x[i : i + batch_size]
    # print(x_batch)
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1) # 统计y_batch矩阵中每行的最大值索引，以数组返回 矩阵的第0维是列方向，第1维是行方向
    # 计算p == t[i : i + batch_size]一一比较后相等的结果个数
    accuracy_cnt += np.sum(p == t[i : i + batch_size]) # 统计比对结果为True的个数

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))  # Accuracy:0.9352
