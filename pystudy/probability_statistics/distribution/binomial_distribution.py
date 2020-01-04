
"""
二项分布图
"""
# from scipy.stats import binom
# import matplotlib.pyplot as plt
# import seaborn
# seaborn.set()
#
# fig, ax = plt.subplots(3, 1)
# params = [(10, 0.25), (10, 0.5), (10, 0.8)]
# x = range(0, 11)
#
# for i in range(len(params)):
#     binom_rv = binom(n=params[i][0], p=params[i][1])
#     ax[i].set_title('n={},p={}'.format(params[i][0], params[i][1]))
#     ax[i].plot(x, binom_rv.pmf(x), 'bo', ms=8)
#     ax[i].vlines(x, 0, binom_rv.pmf(x), colors='b', lw=3)
#     ax[i].set_xlim(0, 10)
#     ax[i].set_ylim(0, 0.35)
#     ax[i].set_xticks(x)
#     ax[i].set_yticks([0, 0.1, 0.2, 0.3])
#
# plt.show()



# from scipy.stats import binom
# import matplotlib.pyplot as plt
# import seaborn
# seaborn.set()
#
# fig, ax = plt.subplots(3, 1)
# params = [(10, 0.25), (10, 0.5), (10, 0.8)]
# x = range(0, 11)
# for i in range(len(params)):
#     binom_rv = binom(n=params[i][0], p=params[i][1])
#     rvs = binom_rv.rvs(size=100000) # 产生服从指定分布的随机数
#     ax[i].hist(rvs, bins=11, normed=True)
#     ax[i].set_title('n={},p={}'.format(params[i][0], params[i][1]))
#     ax[i].set_xlim(0, 10)
#     ax[i].set_ylim(0, 0.4)
#     ax[i].set_xticks(x)
#     print('rvs{}:{}'.format(i, rvs))
#
# plt.show()

# 求二项分布的方差和标准差、期望
import numpy as np
from scipy.stats import binom

binom_rv = binom(n=10, p=0.25)
mean, var, skew, kurt = binom_rv.stats(moments='mvsk')  # 是用函数包中的方法计算的分布的各个理论统计值；

binom_rvs = binom_rv.rvs(size=100000)
E_sim = np.mean(binom_rvs)
S_sim = np.std(binom_rvs)
V_sim = S_sim * S_sim

print('mean={},var={}'.format(mean,var))
print('E_sim={},V_sim={}'.format(E_sim,V_sim))
print('E=np={},V=np(1-p)={}'.format(10 * 0.25,10 * 0.25 * 0.75))    # 期望和方差