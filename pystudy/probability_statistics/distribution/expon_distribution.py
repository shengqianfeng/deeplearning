"""
指数分布
"""
# from scipy.stats import expon
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn
# seaborn.set()
#
# x = np.linspace(0, 10, 1000)
# expon_rv_0 = expon()
# plt.plot(x, expon_rv_0.pdf(x), color='r', lw=5, alpha=0.6, label='`$\\lambda$`=1')
# expon_rv_1 = expon(scale=2)
# plt.plot(x, expon_rv_1.pdf(x), color='b', lw=5, alpha=0.6, label='`$\\lambda$`=0.5')
# plt.legend(loc='best', frameon=False)
#
# plt.show()


# 我们再来对指数型随机变量进行采样生成，我们采样的是服从参数λ=1的指数分布
from scipy.stats import expon
import matplotlib.pyplot as plt
import numpy as np
import seaborn
seaborn.set()

x = np.linspace(0, 10, 1000)
expon_rv = expon()
expon_rvs = expon_rv.rvs(100000)
plt.plot(x, expon_rv.pdf(x), color='r', lw=5, alpha=0.6, label='`$\\lambda$`=1')
plt.hist(expon_rvs, color='b', normed=True, alpha=0.6, bins=50, edgecolor='k')
plt.legend(loc='best', frameon=False)

plt.show()