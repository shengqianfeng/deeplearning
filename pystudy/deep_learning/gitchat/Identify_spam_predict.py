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


output_path = './model/classifier_save/normal/'
model_path = os.path.join(output_path, "model.ckpt")
