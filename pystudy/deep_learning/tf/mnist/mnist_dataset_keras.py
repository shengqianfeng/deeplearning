#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2019/12/28 0028 上午 12:35 
# @Author : jeffsmile 
# @File : mnist_dataset_keras.py
# @desc :使用 Keras 加载 MNIST 数据集
"""
tf.kera.datasets.mnist.load_data(path=‘mnist.npz’)

Arguments:

path: 本地缓存 MNIST 数据集(mnist.npz)的相对路径（~/.keras/datasets）
Returns：

Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
"""
from keras.datasets import mnist
import matplotlib.pyplot as plt

import numpy as np


def load_data(path='mnist.npz'):
    """
    自定义加载数据集
    :param path:
    :return:
    """
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data('mnist.npz')
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

# 绘制前15个手写体数字，以3行5列子图形式展示
fig = plt.figure()
for i in range(15):
    plt.subplot(3, 5, i+1)
    plt.tight_layout() # 自动适配子图尺寸
    plt.imshow(x_train[i], cmap='Greys') # 使用灰色显示像素灰度值
    plt.title("Label: {}".format(y_train[i])) # 设置标签为子图标题
    plt.xticks([]) # 删除x轴标记
    plt.yticks([]) # 删除y轴标记
plt.show()
