#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : random_process.py
@Author : jeffsheng
@Date : 2019/11/14
@Desc : 随机过程：模拟赌徒和庄家的赌博过程
    先采用的样本数也就是赌徒数为 10 个，轮数为 100 轮，也就是每个赌徒最多和庄家对赌 100 轮，
    如果在这个过程中输光了赌本，则提前退出，如果到 100 轮还有赌本，赌局也停止。
"""
import pandas as pd
import random

sample_list = []
# 试验次数
round_num = 100
# 赌徒个数
person_num = 10
for person in range(1, person_num + 1):
    money = 10
    # 每个赌徒进行100次伯努利试验
    for round in range(1, round_num + 1):
        result = random.randint(0, 1)   # 模拟抛硬币
        # 正面：赌徒赢
        if result == 1:
            money = money + 1
        # 反面 庄家赢
        elif result == 0:
            money = money - 1
        # 赌徒没钱了，退出赌局
        if money == 0:
            break
    sample_list.append([person, round, money])

sample_df = pd.DataFrame(sample_list, columns=['person', 'round', 'money'])
sample_df.set_index('person', inplace=True)

print(sample_df)

print("--------------------------")
"""
 金融领域中的一个公式：
            St+1 = St + mu * St * Δt + sigma * St * ϵ * √Δt
实现：利用目前的股价 St 去预测 Δt 时间之后的股价 St+1
"""
import scipy
import matplotlib.pyplot as plt
import seaborn
from math import sqrt
seaborn.set()

# 目前的股价
s0 = 10.0
# 年限
T = 1.0
# 年限的交易日个数
n = 244 * T
# 收益率期望值
mu = 0.15
# 股票的波动率
sigma = 0.2

# 试验次数
n_simulation = 10000

dt = T/n
s_array = []
for i in range(n_simulation):
    s = s0
    # 迭代每个年限内244个交易日后的股价
    for j in range(int(n)):
        # e是一个服从标准正态分布的随机变量
        e = scipy.random.normal()
        s = s + mu*s*dt + sigma*s*e*sqrt(dt)
    s_array.append(s)
# 10000 个对应的一年后股价，然后用柱状图就能看出其总体分布特征
plt.hist(s_array, bins=30, normed=True, edgecolor='k')
plt.show()


print("------------------------------")
"""
不光要计算出股票最终的价格，还有记录下每个Δt 时间点的价格，并把它记录下来
"""
import numpy as np
seaborn.set()

s0 = 10.0
T = 1.0
# 年限内交易日个数
n = 244 * T
mu = 0.15
sigma = 0.2
# 试验次数
n_simulation = 100

dt = T/n
# 生成一个长度为n为0的数组
random_series = np.zeros(int(n), dtype=float)
x = range(0, int(n))
# 进行100次试验
for i in range(n_simulation):
    # 每次进来都把s0作为数组第一个值
    random_series[0] = s0
    # 迭代并生成244个交易日的结果，存入数组中random_series
    for j in range(1, int(n)):
        e = scipy.random.normal()
        random_series[j] = random_series[j-1]+ mu*random_series[j-1]*dt + sigma*random_series[j-1]*e*sqrt(dt)
    # X:x  Y:将数组random_series作为某一个交易日的结果
    plt.plot(x, random_series)

plt.show()
