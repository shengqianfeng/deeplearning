"""
泊松分布
"""
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn
seaborn.set()

fig, ax = plt.subplots(2, 1)
x = range(0, 20)
params = [10, 2]

for i in range(len(params)):
    poisson_rv = poisson(mu=params[i])
    mean, var, skew, kurt = poisson_rv.stats(moments='mvsk')
    ax[i].plot(x, poisson_rv.pmf(x), 'bo', ms=8)
    ax[i].vlines(x, 0, poisson_rv.pmf(x), colors='b', lw=5)
    ax[i].set_title('`$\\lambda$`={}'.format(params[i]))
    ax[i].set_xticks(x)
    print('lambda={},E[X]={},V[X]={}'.format(params[i], mean, var))

plt.show()