#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : ResNet_keras.py
@Author : jeffsheng
@Date : 2020/1/3 0003
@Desc : 残差块
残差块通过跨层的数据通道从而能够训练出有效的深度神经网络
"""
import tensorflow as tf
from tensorflow.keras import layers,activations

net = tf.keras.models.Sequential(
    [tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
    layers.BatchNormalization(), layers.Activation('relu'),
    layers.MaxPool2D(pool_size=3, strides=2, padding='same')])

class Residual(tf.keras.Model):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(num_channels,
                                   padding='same',
                                   kernel_size=3,
                                   strides=strides)
        self.conv2 = layers.Conv2D(num_channels, kernel_size=3,padding='same')
        if use_1x1conv:
            self.conv3 = layers.Conv2D(num_channels,
                                       kernel_size=1,
                                       strides=strides)
        else:
            self.conv3 = None
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()

    def call(self, X):
        Y = activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return activations.relu(Y + X)


def resnet_block(num_channels, num_residuals, first_block=False):
    blk = tf.keras.models.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk

net.add(resnet_block(64, 2, first_block=True))
net.add(resnet_block(128, 2))
net.add(resnet_block(256, 2))
net.add(resnet_block(512, 2))

net.add(layers.GlobalAvgPool2D())
net.add(layers.Dense(10))

# 输出层结构
X = tf.random.uniform(shape=(1,  224, 224 , 1))
for layer in net.layers:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)


# Fashion-MNIST数据集上训练ResNet
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255

net.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(),
              metrics=['accuracy'])

history = net.fit(x_train, y_train,
                    batch_size=64,
                    epochs=5,
                    validation_split=0.2)
test_scores = tf.keras.model.evaluate(x_test, y_test, verbose=2)
