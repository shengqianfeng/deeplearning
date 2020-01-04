# coding: utf-8
import sys, os
# sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle
from pystudy.dataset.mnist import load_mnist
from pystudy.common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

# 读入保存在pickle文件 sample_weight.pkl中的学习到的权重参数
# 这个sample_weight.pkl文件中以字典变量的形式保存了权重和偏置参数
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 输入层---第一个隐藏层
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    # 第一个隐藏层---第二个隐藏层
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    # 第二个隐藏层---输出层
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    # 拿出训练集矩阵每一行元素进行预测
    y = predict(network, x[i])
    # print(y)
    p= np.argmax(y) # 获取概率最高的元素的索引
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))  # Accuracy:0.9352 有93.52 %的数据被正确分类了