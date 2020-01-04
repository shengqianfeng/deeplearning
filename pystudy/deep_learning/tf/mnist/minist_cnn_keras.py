#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2019/12/28 0028 下午 2:08 
# @Author : jeffsmile 
# @File : minist_cnn_keras.py
# @desc :keras实现cnn卷积神经网络
import os
import tensorflow.gfile as gfile
from keras.datasets import mnist
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
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
img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# (60000, 28, 28, 1) <class 'numpy.ndarray'>
print(x_train.shape, type(x_train))
# (10000, 28, 28, 1) <class 'numpy.ndarray'>
print(x_test.shape, type(x_test))
print("--------数据处理：归一化---------------")
# 将数据类型转换为float32
X_train = x_train.astype('float32')
X_test = x_test.astype('float32')
# 数据归一化
X_train /= 255
X_test /= 255
# 60000 train samples
print(X_train.shape[0], 'train samples')
# 10000 test samples
print(X_test.shape[0], 'test samples')
print("---------统计训练数据中各标签数量--------------")
label, count = np.unique(y_train, return_counts=True)
# [0 1 2 3 4 5 6 7 8 9] [5923 6742 5958 6131 5842 5421 5918 6265 5851 5949]
print(label, count)


fig = plt.figure()
plt.bar(label, count, width = 0.7, align='center')
plt.title("Label Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks(label)
plt.ylim(0,7500)

for a,b in zip(label, count):
    plt.text(a, b, '%d' % b, ha='center', va='bottom',fontsize=10)

plt.show()

print("------数据处理：one-hot 编码---------")
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)
Y_test = np_utils.to_categorical(y_test, n_classes)


print("---使用 Keras sequential model 定义 MNIST CNN 网络---------")

model = Sequential()
## Feature Extraction
# 第1层卷积，32个3x3的卷积核 ，激活函数使用 relu
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                 input_shape=input_shape))

# 第2层卷积，64个3x3的卷积核，激活函数使用 relu
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

# 最大池化层，池化窗口 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout 25% 的输入神经元
model.add(Dropout(0.25))

# 将 Pooled feature map 摊平后输入全连接网络
model.add(Flatten())

## Classification
# 全联接层
model.add(Dense(128, activation='relu'))

# Dropout 50% 的输入神经元
model.add(Dropout(0.5))

# 使用 softmax 激活函数做多分类，输出各数字的概率
model.add(Dense(n_classes, activation='softmax'))

print("------查看 MNIST CNN 模型网络结构--------")
model.summary()

for layer in model.layers:
    print(layer.get_output_at(0).get_shape().as_list())

print("----------编译模型----------------")
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
print("---------训练模型，并将指标保存到 history 中------------")
history = model.fit(X_train,
                    Y_train,
                    batch_size=128,
                    epochs=5,
                    verbose=2,
                    validation_data=(X_test, Y_test))
print("---------可视化指标----------")
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
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
