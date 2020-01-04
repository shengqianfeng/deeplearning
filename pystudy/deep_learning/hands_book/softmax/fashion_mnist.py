#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : fashion_mnist.py
@Author : jeffsheng
@Date : 2020/1/2
@Desc : tensorflow加载图像分类数据集fashion-mnist
"""

print("------获取数据集-----------")
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import sys
import matplotlib.pyplot as plt

# 通过keras的dataset包来下载这个数据集,第一次调用会自动从网上获取
from tensorflow.keras.datasets import fashion_mnist
# 训练集中和测试集中的每个类别的图像数分别为6,000和1,000
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(len(x_train),len(x_test))

# 获取第一个样本的图像和标签
feature,label=x_train[0],y_train[0]
# 查看样本特征值形状和类型 (28, 28) uint8
print(feature.shape, feature.dtype)
# 查看样本标签形状和类型 9 <class 'numpy.uint8'> uint8
print(label, type(label), label.dtype)


""""
Fashion-MNIST中一共包括了10个类别，分别为:
    t-shirt（T恤）
    trouser（裤子）
    pullover（套衫）
    dress（连衣裙）
    coat（外套）
    sandal（凉鞋）
    shirt（衬衫）
    sneaker（运动鞋）
    bag（包）
    ankle boot（短靴）
以下函数可以将数值标签转成相应的文本标签
"""
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 在一行里画出多张图像和对应标签
def show_fashion_mnist(images, labels):
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)))
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


# 看一下训练数据集中前9个样本的图像内容和文本标签
X, y = [], []
for i in range(10):
    X.append(x_train[i])
    y.append(y_train[i])
show_fashion_mnist(X, get_fashion_mnist_labels(y))


print("----------读取小批量数据--------------")
batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4
train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(256)

# 查看读取一遍训练数据需要的时间
start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))
