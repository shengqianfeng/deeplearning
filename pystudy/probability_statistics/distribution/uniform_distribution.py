"""
均匀分布
"""
# from scipy.stats import uniform
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn
# seaborn.set()
#
# x = np.linspace(-1, 3.5, 1000)
# uniform_rv_0 = uniform()
# uniform_rv_1 = uniform(loc=0.5, scale=2)
#
# plt.plot(x, uniform_rv_0.pdf(x), color='r', lw=5, alpha=0.6, label='[0,1]')
# plt.plot(x, uniform_rv_1.pdf(x), color='b', lw=5, alpha=0.6, label='[0.5,2.5]')
# plt.legend(loc='best', frameon=False)
#
# plt.show()


"""
均匀分布的采样观测
"""
from scipy.stats import uniform
import matplotlib.pyplot as plt
import numpy as np
import seaborn
seaborn.set()

x = np.linspace(0, 3.5, 1000)
uniform_rv = uniform(1, 2)
uniform_rvs = uniform_rv.rvs(100000)
plt.plot(x, uniform_rv.pdf(x), 'r-', lw=5, alpha=0.6, label='[1,3]')
plt.hist(uniform_rvs, color='b', normed=True, alpha=0.6, bins=50, edgecolor='k')
plt.legend(loc='best', frameon=False)

plt.show()