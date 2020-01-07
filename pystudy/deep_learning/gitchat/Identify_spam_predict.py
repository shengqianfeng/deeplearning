#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : Identify_spam_predict.py
@Author : jeffsheng
@Date : 2020/1/7
@Desc : rnn识别垃圾邮件预测
"""
import os
import tensorflow as tf
from  pystudy.deep_learning.gitchat.rcnn_model import rcnn
import  numpy as np


output_path = './model/classifier_save/normal/'
model_path = os.path.join(output_path, "model.ckpt")
sess = tf.Session()
net = rcnn(sess,model_path)
text_data_target,text_data_train = net.load_data()
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(net.max_sequence_length, min_frequency=net.min_word_frequency)
text_processed = np.array(list(vocab_processor.fit_transform(text_data_train)))
# 词向量的大小
vocab_size = len(vocab_processor.vocabulary_)
print("Vocabulary Size: {:d}".format(vocab_size))

# shuffle，可以打乱数据行序，使数据随机化
text_processed = np.array(text_processed)
# 将text_data_target转为0和1的结果标签
text_data_target = np.array([1 if x == 'ham' else 0 for x in text_data_target])
'''
shuffle与permutation的区别
    1函数shuffle与permutation都是对原来的数组进行重新洗牌（即随机打乱原来的元素顺序）；
    2区别在于shuffle直接在原来的数组上进行操作，改变原来数组的顺序，无返回值。
    3而permutation不直接在原来的数组上进行操作，而是返回一个新的打乱顺序的数组，并不改变原来的数组
'''
shuffled_ix = np.random.permutation(np.arange(len(text_data_target)))
# 按照随机打乱词向量及标签的顺序重新返回数据集和训练集
x_shuffled = text_processed[shuffled_ix]
y_shuffled = text_data_target[shuffled_ix]

'''
shuffle 数据后，将数据集分为 80% 训练集和 20% 测试集 
如果想做交叉验证 cross-validation ，可以将 测试集 进一步分为测试集和验证集来调参
'''
ix_cutoff = int(len(y_shuffled)*0.80)
# 分割数据：训练集和测试集
x_train, x_test = x_shuffled[:ix_cutoff], x_shuffled[ix_cutoff:]
# 分割标签：训练集标签和测试集标签
y_train, y_test = y_shuffled[:ix_cutoff], y_shuffled[ix_cutoff:]
x_data = tf.placeholder(tf.int32, [None, net.max_sequence_length])
embedding_mat = tf.Variable(tf.random_uniform([vocab_size, net.embedding_size], -1.0, 1.0))
embedding_output = tf.nn.embedding_lookup(embedding_mat, x_data)
print(net.predict(embedding_output,x_train))
