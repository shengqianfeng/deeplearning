# coding: utf-8
import numpy as np
import pickle
from pystudy.dataset.mnist import load_mnist
from pystudy.common.functions import sigmoid, softmax

"""
神经网络复盘之演进：三层神经网络mnist训练集
输入层：784个神经元
输出层：10个神经元
隐藏层1:50个神经元
隐藏层2:100个神经元

演示：读入事先训练好的参数加载模型，预测测试集并计算准确率
"""

def get_data():
    """
    正规化：将 normalize设置成 True后，函数内部会进行转换，将图像的各个像素值除以255，使得数据的值在0.0～1.0的范围内
    :return:
    """
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    """
     读入保存在ample_weight.pkl中的学习到的权重参数
     这个sample_weight.pkl文件中以字典变量的形式保存了权重和偏置参数
    """
    with open("../dataset/sample_weight.pkl", 'rb') as f:
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
    # 拿出训练集矩阵每一行元素(也就是每一张图片)进行预测，输出的是长度为10的一维概率矩阵
    y = predict(network, x[i])
    # print(y)
    p= np.argmax(y) # 获取概率最高的元素的索引
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))  # Accuracy:0.9352 有93.52 %的数据被正确分类了