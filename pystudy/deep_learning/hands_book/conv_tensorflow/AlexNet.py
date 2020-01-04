#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : AlexNet.py
@Author : jeffsheng
@Date : 2020/1/3 0003
@Desc : AlexNet
1 数据和硬件的发展催生了AlexNet
2 2012年，Alex发明的AlexNet横空出世，AlexNet使用了8层卷积神经网络，
并以很大的优势赢得了ImageNet 2012图像识别挑战赛。

结论：
1 AlexNet跟LeNet结构类似，但使用了更多的卷积层和更大的参数空间来拟合大规模数据集ImageNet。它是浅层神经网络和深度神经网络的分界线
2 虽然看上去AlexNet的实现比LeNet的实现也就多了几行代码而已，但这个观念上的转变和真正优秀实验结果的产生令学术界付出了很多年。
"""

import tensorflow as tf
print(tf.__version__)
# 建议采用GPU进行训练，需要使用tensorflow-gpu-2.0并设置memory_growth
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

# AlexNet有5层卷积和2层全连接隐藏层，以及1个全连接输出层
net = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=96,kernel_size=11,strides=4,activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
    tf.keras.layers.Conv2D(filters=256,kernel_size=5,padding='same',activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
    tf.keras.layers.Conv2D(filters=384,kernel_size=3,padding='same',activation='relu'),
    tf.keras.layers.Conv2D(filters=384,kernel_size=3,padding='same',activation='relu'),
    tf.keras.layers.Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4096,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10,activation='sigmoid')
])

# 构造一个高和宽均为224的单通道数据样本来观察每一层的输出形状
X = tf.random.uniform((1,224,224,1))
for layer in net.layers:
    X = layer(X)
    print(layer.name, 'output shape\t', X.shape)

"""
conv2d output shape	 (1, 54, 54, 96)
max_pooling2d output shape	 (1, 26, 26, 96)
conv2d_1 output shape	 (1, 26, 26, 256)
max_pooling2d_1 output shape	 (1, 12, 12, 256)
conv2d_2 output shape	 (1, 12, 12, 384)
conv2d_3 output shape	 (1, 12, 12, 384)
conv2d_4 output shape	 (1, 12, 12, 256)
max_pooling2d_2 output shape	 (1, 5, 5, 256)
flatten output shape	 (1, 6400)
dense output shape	 (1, 4096)
dropout output shape	 (1, 4096)
dense_1 output shape	 (1, 4096)
dropout_1 output shape	 (1, 4096)
dense_2 output shape	 (1, 10)
"""
print("----------------读取数据-----------------------------")
"""
虽然论文中AlexNet使用ImageNet数据集，但因为ImageNet数据集训练时间较长，我们仍用Fashion-MNIST数据集来演示AlexNet
"""
# 读取数据的时候我们额外做了一步将图像高和宽扩大到AlexNet使用的图像高和宽224
import numpy as np

class DataLoader():
    def __init__(self):
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()
        self.train_images = np.expand_dims(self.train_images.astype(np.float32)/255.0,axis=-1)
        self.test_images = np.expand_dims(self.test_images.astype(np.float32)/255.0,axis=-1)
        self.train_labels = self.train_labels.astype(np.int32)
        self.test_labels = self.test_labels.astype(np.int32)
        # 获取训练集和测试集样本大小
        self.num_train, self.num_test = self.train_images.shape[0], self.test_images.shape[0]
    # 从60000中抽取128个下标位置
    def get_batch_train(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_images)[0], batch_size)
        #need to resize images to (224,224)  将抽取的128个样本宽高填充至224
        resized_images = tf.image.resize_with_pad(self.train_images[index], 224, 224,)
        return resized_images.numpy(), self.train_labels[index]

    def get_batch_test(self, batch_size):
        index = np.random.randint(0, np.shape(self.test_images)[0], batch_size)
        #need to resize images to (224,224)
        resized_images = tf.image.resize_with_pad(self.test_images[index],224,224,)
        return resized_images.numpy(), self.test_labels[index]

batch_size = 128
dataLoader = DataLoader()
x_batch, y_batch = dataLoader.get_batch_train(batch_size)
# x_batch shape: (128, 224, 224, 1) y_batch shape: (128,)
print("x_batch shape:",x_batch.shape,"y_batch shape:", y_batch.shape)

print("--------------------------------训练AlexNet-----------------------------")
def train_alexnet():
    epoch = 5
    num_iter = dataLoader.num_train//batch_size
    for e in range(epoch):
        for n in range(num_iter):
            x_batch, y_batch = dataLoader.get_batch_train(batch_size)
            net.fit(x_batch, y_batch)
            if n%20 == 0:
                net.save_weights("5.6_alexnet_weights.h5")

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)

net.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

x_batch, y_batch = dataLoader.get_batch_train(batch_size)
net.fit(x_batch, y_batch)
# 需要进行训练，请执行train_alexnet()函数
# train_alexnet()

# 将训练好的参数读入，然后取测试数据计算测试准确率
net.load_weights("5.6_alexnet_weights.h5")

x_test, y_test = dataLoader.get_batch_test(2000)
net.evaluate(x_test, y_test, verbose=2)
