#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : hmm_viterbi_decode.py
@Author : jeffsheng
@Date : 2020/2/22 0022
@Desc : 隐马尔科夫链的观测序列解码，输出概率最大的隐含状态序列
"""

import numpy as np
from hmmlearn import hmm

# 隐状态集合Q
states = ['box1', 'box2', 'box3']
# 观测集合V
observations = ['black', 'white']
# 初始概率pi
start_probability = np.array([0.3, 0.5, 0.2])
# 状态转移矩阵A
transition_probability = np.array([
  [0.4, 0.4, 0.2],
  [0.3, 0.2, 0.5],
  [0.2, 0.6, 0.2]
])

# 观测概率矩阵B
emission_probability = np.array([
  [0.2, 0.8],
  [0.6, 0.4],
  [0.4, 0.6]
])

# 选用MultinomialHMM对离散观测状态建模
model = hmm.MultinomialHMM(n_components=len(states))
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

# 观测序列
obervation_list = np.array([0, 1, 0])
# 调用维特比算法对观测序列进行隐含状态解码
logprob, box_list = model.decode(obervation_list.reshape(-1, 1), algorithm='viterbi')
# 输出解码的隐含状态序列
print(box_list)
for i in range(len(obervation_list)):
    print(states[box_list[i]])


