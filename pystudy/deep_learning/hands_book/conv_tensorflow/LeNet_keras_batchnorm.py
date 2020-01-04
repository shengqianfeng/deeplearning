#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : LeNet_keras_batchnorm.py
@Author : jeffsheng
@Date : 2020/1/3 0003
@Desc : 用keras实现使用批量归一化的LeNet
挑战：
    对深层神经网络来说，即使输入数据已做标准化，训练中模型参数的更新依然很容易
造成靠近输出层输出的剧烈变化。这种计算数值的不稳定性通常令我们难以训练出有效的深度模型

解决办法：
    批量归一化的提出正是为了应对深度模型训练的挑战。在模型训练时，批量归一化利用小批量上的均值和标准差，不断调整神经网络中间输出，从而使整个神经网络在各层的中间输出的数值更稳定

"""
import tensorflow as tf


net = tf.keras.models.Sequential()
net.add(tf.keras.layers.Conv2D(filters=6,kernel_size=5))
net.add(tf.keras.layers.BatchNormalization())
net.add(tf.keras.layers.Activation('sigmoid'))
net.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
net.add(tf.keras.layers.Conv2D(filters=16,kernel_size=5))
net.add(tf.keras.layers.BatchNormalization())
net.add(tf.keras.layers.Activation('sigmoid'))
net.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
net.add(tf.keras.layers.Flatten())
net.add(tf.keras.layers.Dense(120))
net.add(tf.keras.layers.BatchNormalization())
net.add(tf.keras.layers.Activation('sigmoid'))
net.add(tf.keras.layers.Dense(84))
net.add(tf.keras.layers.BatchNormalization())
net.add(tf.keras.layers.Activation('sigmoid'))
net.add(tf.keras.layers.Dense(10,activation='sigmoid'))


# 使用跟LeNet同样的超参数进行训练
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255

net.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(),
              metrics=['accuracy'])
history = net.fit(x_train, y_train,
                    batch_size=64,
                    epochs=5,
                    validation_split=0.2)
test_scores = net.evaluate(x_test, y_test, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])



