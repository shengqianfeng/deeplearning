#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : markov_steady_state.py
@Author : jeffsheng
@Date : 2019/11/18
@Desc : 马尔科夫稳态
"""

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn
# seaborn.set()
#
# # 状态转移矩阵
# transfer_matrix = np.array([[0.7, 0.1, 0.2],
#                 [0.3, 0.5, 0.2],
#                 [0.1, 0.3, 0.6]], dtype='float32')
#
# # 初始状态分布数组
# start_state_array = np.array([[0.50, 0.30, 0.20],
#                   [0.13, 0.28, 0.59],
#                   [0.10, 0.85, 0.05]], dtype='float32')
#
# trans_step = 10
#
# for i in range(3):
#     state_1_value = []
#     state_2_value = []
#     state_3_value = []
#     for _ in range(trans_step):
#         start_state_array[i] = np.dot(start_state_array[i], transfer_matrix)
#         state_1_value.append(start_state_array[i][0])
#         state_2_value.append(start_state_array[i][1])
#         state_3_value.append(start_state_array[i][2])
#
#     x = np.arange(trans_step)
#     plt.plot(x, state_1_value, label='state_1')
#     plt.plot(x, state_2_value, label='state_2')
#     plt.plot(x, state_3_value, label='state_3')
#     plt.legend()
#
#     print(start_state_array[i])
#
# plt.gca().axes.set_xticks(np.arange(0, trans_step))
# plt.gca().axes.set_yticks(np.arange(0.2, 0.6, 0.05))
# plt.gca().axes.set_xlabel('n step')
# plt.gca().axes.set_ylabel('p')
#
# plt.show()

"""
基于马尔科夫链的采样方法对目标分布进行采样，统计最终各个状态中样本的实际比例
"""
import numpy as np
from scipy.stats import uniform
import random

def randomstate_gen(cur_state, transfer_matrix):
    uniform_rvs = uniform().rvs(1) # 生成一个0-1的均匀分布的随机数
    i = cur_state-1
    # 根据随机数落入的区间来判断下一步的状态转移
    if uniform_rvs[0] <= transfer_matrix[i][0]:
        return 1
    elif uniform_rvs[0] <= transfer_matrix[i][0] + transfer_matrix[i][1]:
        return 2
    else:
        return 3


transfer_matrix = np.array([[0.7, 0.1, 0.2],
                            [0.3, 0.5, 0.2],
                            [0.1, 0.3, 0.6]], dtype='float32')
m = 10000
N = 100000

cur_state = random.choice([1, 2, 3])
state_list = []
for i in range(m+N):
    state_list.append(cur_state)
    cur_state = randomstate_gen(cur_state, transfer_matrix)

state_list = state_list[m:]

print(state_list.count(1)/float(len(state_list)))
print(state_list.count(2)/float(len(state_list)))
print(state_list.count(3)/float(len(state_list)))