#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : gru_impl_keras.py
@Author : jeffsheng
@Date : 2020/2/13 0013
@Desc : keras来实现门控神经单元gru
"""
import tensorflow as tf
from tensorflow import keras
import time
import math
import numpy as np
import sys
sys.path.append("..")
import pystudy.deep_learning.hands_book.d2lzh_tensorflow2 as d2l
(corpus_indices, char_to_idx, idx_to_char,vocab_size) = d2l.load_data_jay_lyrics()

num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']
print("--------初始化模型参数----------")
# 超参数num_hiddens定义了隐藏单元的个数
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size


gru_layer = keras.layers.GRU(num_hiddens,time_major=True,return_sequences=True,return_state=True)
model = d2l.RNNModel(gru_layer, vocab_size)
d2l.train_and_predict_rnn_keras(model, num_hiddens, vocab_size,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)





