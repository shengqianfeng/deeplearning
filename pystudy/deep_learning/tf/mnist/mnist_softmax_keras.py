#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2019/12/28 0028 上午 11:23 
# @Author : jeffsmile 
# @File : mnist_softmax_keras.py
# @desc :keras实现全连接层的神经网络


from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import os
import tensorflow.gfile as gfile
from keras.models import load_model


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

print("--------加载 MNIST 数据集---------------")
(x_train, y_train), (x_test, y_test) = load_data('mnist.npz')
# (60000, 28, 28) <class 'numpy.ndarray'>
print(x_train.shape, type(x_train))
# (60000,) <class 'numpy.ndarray'>
print(y_train.shape, type(y_train))
print("--------数据处理：规范化---------------")
# 将图像本身从[28,28]转换为[784,]
X_train = x_train.reshape(60000, 784)
X_test = x_test.reshape(10000, 784)
# (60000, 784) <class 'numpy.ndarray'>
print(X_train.shape, type(X_train))
# (10000, 784) <class 'numpy.ndarray'>
print(X_test.shape, type(X_test))
print("--------数据处理：归一化---------------")
# 将数据类型转换为float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# 数据归一化
X_train /= 255
X_test /= 255
print("---------统计训练数据中各标签数量--------------")
label, count = np.unique(y_train, return_counts=True)
# [0 1 2 3 4 5 6 7 8 9]  [5923 6742 5958 6131 5842 5421 5918 6265 5851 5949]
print(label, count)

fig = plt.figure()
# 柱状图  x轴为label，y轴为count
plt.bar(label, count, width = 0.7, align='center')
# 柱状图标题
plt.title("Label Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
# x轴数据为label数组
plt.xticks(label)
plt.ylim(0,7500)
# 在柱状图上标注出标签数量count
for a,b in zip(label, count):
    plt.text(a, b, '%d' % b, ha='center', va='bottom', fontsize=10)

plt.show()

print("------数据处理：one-hot 编码---------")
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)
Y_test = np_utils.to_categorical(y_test, n_classes)
# 5
print(y_train[0])
# [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
print(Y_train[0])

print("--------使用 Keras sequential model 定义神经网络--------")
model = Sequential()
# 定义512个神经元的全连接层，指定输入数据为一维784
model.add(Dense(512, input_shape=(784,)))
# 激活函数relu
model.add(Activation('relu'))

# 定义第二个512神经元的全连接层，无需指定输入尺寸默认以上一层输出为输入
model.add(Dense(512))
# 激活函数relu
model.add(Activation('relu'))

# 定义10个神经元的全连接层（输出层）
model.add(Dense(10))
# 激活函数softmax，每个神经元输出概率值
model.add(Activation('softmax'))

print("----------编译模型----------------")
# 编译模型，损失函数为交叉熵，统计准确率，梯度下降的优化器adam
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
print("---------训练模型，并将指标保存到 history 中------------")
history = model.fit(X_train,
                    Y_train,
                    batch_size=128,
                    epochs=5,
                    verbose=2,# 日志复杂度级别
                    validation_data=(X_test, Y_test))


print("---------可视化指标----------")
history_dict = history.history
fig = plt.figure()
# 绘制2行一列的第一个子图  精确值随着epoch的变化
plt.subplot(2,1,1)
plt.plot(history_dict['accuracy'])
plt.plot(history_dict['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
# 图像右下角放置标签提示
plt.legend(['train', 'test'], loc='lower right')

# 绘制2行一列的第二个子图 损失函数值随着epoch的变化
plt.subplot(2,1,2)
plt.plot(history_dict['loss'])
plt.plot(history_dict['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()

plt.show()

print("---------keras保存模型为hdf5文件：包含权重参数、损失函数、优化器等数据流图信息----------------")
save_dir = "./model/"
if gfile.Exists(save_dir):
    gfile.DeleteRecursively(save_dir)
gfile.MakeDirs(save_dir)

model_name = 'keras_mnist.h5'
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

print("-------keras加载模型hdf5文件----------")
mnist_model = load_model(model_path)
print("--------统计模型在测试集上的分类结果------------")
loss_and_metrics = mnist_model.evaluate(X_test, Y_test, verbose=2)

print("Test Loss: {}".format(loss_and_metrics[0]))
print("Test Accuracy: {}%".format(loss_and_metrics[1] * 100))

predicted_classes = mnist_model.predict_classes(X_test)

# 返回y_test预测正确的下标
correct_indices = np.nonzero(predicted_classes == y_test)[0]
# 返回y_test预测不正确的下标
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
print("Classified correctly count: {}".format(len(correct_indices)))
print("Classified incorrectly count: {}".format(len(incorrect_indices)))
