#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2019/12/28 0028 下午 5:19 
# @Author : jeffsmile 
# @File : training_analysis.py
# @desc :训练过程分析
# 需要实现训练各种优化器的模型：adadelta、sgd、adagrad、adam、rmsprop

import glob
import pickle

import matplotlib.pyplot as plt

print("----------加载训练过程记录-----------")
history_file = './history/train_demo/captcha_adam_binary_crossentropy_bs_100_epochs_10.history'
with open(history_file, 'rb') as f:
    history = pickle.load(f)

print("----------训练过程可视化------------")
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()

plt.show()

HISTORY_DIR = './history/train_demo/'
HISTORY_FORMAT = '.history'

# 定义过程可视化方法
def plot_training(history=None, metric='accuracy', title='Model Accuracy', loc='lower right'):
    model_list = []
    fig = plt.figure(figsize=(10, 8))
    for key, val in history.items():
        model_list.append(key.replace(HISTORY_DIR, '').rstrip('.history'))
        plt.plot(val[metric])

    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(model_list, loc=loc)
    plt.show()


print("---------加载预训练模型记录--------------")
history = {}
for filename in glob.glob(HISTORY_DIR + '*.history'):
    with open(filename, 'rb') as f:
        history[filename] = pickle.load(f)
for key, val in history.items():
    print(key.replace(HISTORY_DIR, '').rstrip('.history'), val.keys())

print("---------准确率变化（训练集）---------")
plot_training(history)
print("----------损失值变化（训练集）-------------")
plot_training(history, metric='loss', title='Model Loss', loc='upper right')

print("---------准确率变化（测试集）---------")
plot_training(history, metric='val_accuracy', title='Model Accuracy (val)')
print("----------损失值变化（测试集）-------------")
plot_training(history, metric='val_loss', title='Model Loss (val)', loc='upper right')
