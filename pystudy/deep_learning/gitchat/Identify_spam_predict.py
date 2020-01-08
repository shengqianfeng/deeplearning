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
inputs = tf.placeholder(tf.float64, [None, 25, 1])
print(net.predict(inputs))
