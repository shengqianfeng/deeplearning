#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : gru_impl_from_zero.py
@Author : jeffsheng
@Date : 2020/2/13 0013
@Desc : 如何从零开始实现门控循环单元(gru,gated recurrent unit)
"""


import tensorflow as tf
from tensorflow import keras
import time
import math
import numpy as np
import sys
sys.path.append("..")
import pystudy.deep_learning.hands_book.d2lzh_tensorflow2 as d2l


print("--------读取数据集--------")
# 依然使用周杰伦歌词数据集来训练模型作词
(corpus_indices, char_to_idx, idx_to_char,vocab_size) = d2l.load_data_jay_lyrics()


print("--------初始化模型参数----------")
# 超参数num_hiddens定义了隐藏单元的个数
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size

def get_params():

    def _one(shape):
        return tf.Variable(tf.random.normal(shape=shape,stddev=0.01,mean=0,dtype=tf.float32))

    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                tf.Variable(tf.zeros(num_hiddens), dtype=tf.float32))

    W_xz, W_hz, b_z = _three()  # 更新门参数
    W_xr, W_hr, b_r = _three()  # 重置门参数
    W_xh, W_hh, b_h = _three()  # 候选隐藏状态参数
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = tf.Variable(tf.zeros(num_outputs), dtype=tf.float32)
    # 附上梯度
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]

    return params



# 定义隐藏状态初始化函数init_gru_state
def init_gru_state(batch_size, num_hiddens):
    # 返回由一个形状为(批量大小, 隐藏单元个数)的值为0的Tensor组成的元组
    return (tf.zeros(shape=(batch_size, num_hiddens)), )


# 根据门控循环单元的计算表达式定义模型
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        X=tf.reshape(X,[-1,W_xh.shape[0]])
        Z = tf.sigmoid(tf.matmul(X, W_xz) + tf.matmul(H, W_hz) + b_z)
        R = tf.sigmoid(tf.matmul(X, W_xr) + tf.matmul(H, W_hr) + b_r)
        H_tilda = tf.tanh(tf.matmul(X, W_xh) + tf.matmul(R * H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = tf.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)


print("--------训练模型并创作歌词--------------")
num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

"""
我们在训练模型时只使用相邻采样。
设置好超参数后，我们将训练模型并根据前缀“分开”和“不分开”分别创作长度为50个字符的一段歌词
"""
# 每过40个迭代周期便根据当前训练的模型创作一段歌词
d2l.train_and_predict_rnn(gru, get_params, init_gru_state, num_hiddens,
                          vocab_size, corpus_indices, idx_to_char,
                          char_to_idx, False, num_epochs, num_steps,
                          lr,clipping_theta, batch_size, pred_period,
                          pred_len,prefixes)



