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
from pystudy.deep_learning.gitchat.rcnn_model import rcnn
import numpy as np
from pystudy.deep_learning.gitchat.Identity_spam_intent_recognize import recognize

output_path = './model/classifier_save/normal/'
model_path = os.path.join(output_path, "model.ckpt")
net = rcnn(model_path)

text_data_target,text_data_train = net.load_data()
text_data_target = np.array([1 if x == 'ham' else 0 for x in text_data_target])
print("-----数据加载完成-----")
# ix_cutoff = int(len(text_data_target)*0.80)
# x_train, y_train = text_data_train[ix_cutoff:], text_data_target[ix_cutoff:]

spamRecognize = recognize(net)
spamRecognize.recognize(text_data_train, text_data_target)

