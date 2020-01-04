#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : LeNet.py
@Author : jeffsheng
@Date : 2020/1/3 0003
@Desc : 卷积神经网络（LeNet）
    LeNet展示了通过梯度下降训练卷积神经网络可以达到手写数字识别在当时最先进的结果。
    这个奠基性的工作第一次将卷积神经网络推上舞台，为世人所知


相对于含单隐藏层的多层感知机模型的缺陷：
1 图像在同一列邻近的像素在这个向量中可能相距较远。它们构成的模式可能难以被模型识别
2 对于大尺寸的输入图像，使用全连接层容易导致模型过大。这会带来过于复杂的模型和过高的存储开销。

卷积层试图解决的两个问题：
1 卷积层保留输入形状，使图像的像素在高和宽两个方向上的相关性均可能被有效识别；
2 卷积层通过滑动窗口将同一卷积核与不同位置的输入重复计算，从而避免参数尺寸过大
"""

import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6,kernel_size=5, activation='sigmoid',input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(filters=16,kernel_size=5,activation='sigmoid'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120,activation='sigmoid'),
    tf.keras.layers.Dense(84,activation='sigmoid'),
    tf.keras.layers.Dense(10,activation='sigmoid')
])

# 构造一个高和宽均为28的单通道数据样本，并逐层进行前向计算来查看每个层的输出形状
X = tf.random.uniform((1,28,28,1))
for layer in net.layers:
    X = layer(X)
    print(layer.name, 'output shape\t', X.shape)

"""
conv2d output shape	 (1, 24, 24, 6)             --->(1,28,28,1)输入数据经过第一层卷积后为(1,24,24,6)
max_pooling2d output shape	 (1, 12, 12, 6)     --->(1,24,24,6)最大池化后减小一半为（1,12,12,6）
conv2d_1 output shape	 (1, 8, 8, 16)          --->（1,12,12,6）输入第二层卷积层后为(1, 8, 8, 16) ,注意：多通道输入经过滤波器后变为二维
max_pooling2d_1 output shape	 (1, 4, 4, 16)  --->(1, 8, 8, 16)最大池化后宽高减小一半（1，4,4,16） 
flatten output shape	 (1, 256)               --->输入全连接层前要摊平：(1,256)
dense output shape	 (1, 120)                   
dense_1 output shape	 (1, 84)
dense_2 output shape	 (1, 10)
"""

print("----------获取数据和训练模型---------")
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = tf.reshape(train_images, (train_images.shape[0],train_images.shape[1],train_images.shape[2], 1))
print(train_images.shape)

test_images = tf.reshape(test_images, (test_images.shape[0],test_images.shape[1],test_images.shape[2], 1))


# 损失函数和训练算法依然采用交叉熵损失函数(cross entropy)和小批量随机梯度下降(SGD)
# nesterov:布尔值，确定是否使用Nesterov动量
optimizer = tf.keras.optimizers.SGD(learning_rate=0.9, momentum=0.0, nesterov=False)

net.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 模型的fit函数有两个参数:
# 1 shuffle用于将数据打乱
# 2 validation_split用于在没有提供验证集的时候，按一定比例从训练集中取出一部分作为验证集
net.fit(train_images, train_labels, epochs=5, validation_split=0.1)

net.evaluate(test_images, test_labels, verbose=2)


