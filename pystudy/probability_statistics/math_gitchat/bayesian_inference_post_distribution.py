#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : bayesian_inference_post_distribution.py
@Author : jeffsheng
@Date : 2020/2/20 0020
@Desc : 贝叶斯统计推断计算最后的后验分布
根据：先验分布+观测数据=后验分布
step1 选取最终的几组α和β组合作为beta的先验分布参数作为最终贝叶斯公式的带入参数进行计算。
step2 选择几组观测数据，利用贝叶斯统计推断来得到后验分布


beta分布于二项分布的共轭性：
    先验分布是beta 分布，观测数据服从二项分布，得到的后验仍然是beta 分布，
也就是说beta 分布是二项分布的共轭先验，即将先验beta 分布与
二项分布组合在一起之后，得到的后验分布与先验分布的表达式形式仍然是一样的。
除此之外，正态分布也是自身的共轭先验
"""

from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np
import seaborn

seaborn.set()

theta_real = 0.62
# 观测数据-试验次数
n_array = [5, 10, 20, 100, 500, 1000]
# 观测数据-事件发生次数（比如正面朝上的次数）
y_array = [2, 4, 11, 60, 306, 614]
# 选取最终的几组α和β  ，分别代表均匀分布、U型分布、类似正态分布，代表了我们对θ的不同先验认知，
# 它们概率取值高低跟自身分布形式有关

# 观测结果:
# 1 随着观测数据的增多，后验分布会越来越集中，表示对参数的确定性也越高。
# 2 当观测数据足够多时，不同先验分布对应的后验分布都会收敛于同一个值，
# 数据越多，通过最大后验准则MAP得到的估计值就跟参数的实际值越接近。

"""
MAP-最大后验准则
一般而言我们都会选择后验分布概率密度函数曲线的峰值作为我们最终对于未知参数的估计值。
这就是贝叶斯推断中的最大后验概率MAP准则，即选择在一个给定数据下，具有最大后验概率的值。

"""
beta_params = [(0.25, 0.25), (1, 1), (10, 10)]
x = np.linspace(0, 1, 100)

fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)

for i in range(2):
    for j in range(3):
        n = n_array[3 * i + j]
        y = y_array[3 * i + j]
        for (a_prior, b_prior), c in zip(beta_params, ('b', 'r', 'g')):
            # 更新α和β
            a_post = a_prior + y
            b_post = b_prior + n - y
            # 代入α和β继续生成新的beta分布
            p_theta_given_y = beta.pdf(x, a_post, b_post)
            ax[i, j].plot(x, p_theta_given_y, c)
            ax[i, j].fill_between(x, 0, p_theta_given_y, color=c, alpha=0.25)
        # K: black黑色  现实中的实际概率值
        ax[i, j].axvline(theta_real, ymax=0.5, color='k')
        ax[i, j].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax[i, j].set_title('n={},y={}'.format(n, y))

ax[0, 0].set_ylabel('`$p(\\theta|y)$`')
ax[1, 0].set_ylabel('`$p(\\theta|y)$`')
ax[1, 1].set_xlabel('`$\\theta$`')
plt.show()




