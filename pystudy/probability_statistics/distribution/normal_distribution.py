
"""
正态分布，又叫高斯分布
观察两组正态分布的概率密度函数取值
一组是均值为0，标准差为1
另一组，我们取均值为1，标准差为2
均值用参数loc 来描述，方差用参数scale 来描述
"""
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import seaborn
seaborn.set()

fig, ax = plt.subplots(1, 1)
norm_0 = norm(loc=0, scale=1)# 均值为0，标准差为1
norm_1 = norm(loc=1, scale=2)# 均值为1，标准差为2

x = np.linspace(-10, 10, 1000)
ax.plot(x, norm_0.pdf(x), color='red', lw=5, alpha=0.6, label='loc=0, scale=1')
ax.plot(x, norm_1.pdf(x), color='blue', lw=5, alpha=0.6, label='loc=1, scale=2')
ax.legend(loc='best', frameon=False)

plt.show()


# 基于指定分布的重复采样，来观察和验证模拟试验的情况
seaborn.set()

norm_rv = norm(loc=2, scale=2)# 均值为2，标准差为2
norm_rvs = norm_rv.rvs(size=100000)# 根据概率分布，返回随机数
x = np.linspace(-10, 10, 1000)
plt.plot(x, norm_rv.pdf(x), 'r', lw=5, alpha=0.6, label="`$\\mu$=2,$\\sigma=2$`")# pdf()概率密度函数
plt.hist(norm_rvs, normed=True, bins=50, alpha=0.6, edgecolor='k')
plt.legend()
plt.show()