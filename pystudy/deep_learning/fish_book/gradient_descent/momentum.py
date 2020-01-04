
from mxnet import nd
"""
动量法-梯度下降

"""
import pystudy.nn_study.d2lzh as d2l

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2


"""
初始值v1和v2均为0，经过一次梯度更新后v1和v2变为
    v1 = gamma * v1 + eta * 0.2 * x1
    v2 = gamma * v2 + eta * 4 * x2
的结果值，也就是：
    v1 = eta * 0.2 * x1
    v2 = eta * 4 * x2
也就是第一次更新为sgd方式的梯度下降。
以后的每一次v1和v2都有了不为0的值，v1和v2大于0则下降加速，v1和v2小于0则下降减速
"""
def momentum_2d(x1, x2, v1, v2):
    v1 = gamma * v1 + eta * 0.2 * x1
    v2 = gamma * v2 + eta * 4 * x2
    return x1 - v1, x2 - v2, v1, v2
#eta为学习率， gamma为动量超参数
eta, gamma = 0.4, 0.5
# 可以看到使用较小的学习率 η=0.4和动量超参数γ=0.5 时，
# 动量法在竖直方向上的移动更加平滑，且在水平方向上更快逼近最优解
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
d2l.plt.show()

# 较大的学习率 η=0.6，gamma=0.5此时自变量也不再发散
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
d2l.plt.show()


# features (1503,5)  labels(1,1503)
features, labels = d2l.get_data_ch7()

def init_momentum_states():
    v_w = nd.zeros((features.shape[1], 1))# (5,1)
    v_b = nd.zeros(1)
    return (v_w, v_b)

def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v[:] = hyperparams['momentum'] * v + hyperparams['lr'] * p.grad
        p[:] -= v

d2l.train_ch7(sgd_momentum, init_momentum_states(), {'lr': 0.02, 'momentum': 0.5}, features, labels)
d2l.plt.show()

# 保持学习率不变
# 将动量超参数momentum增大到0.9
d2l.train_ch7(sgd_momentum, init_momentum_states(),
              {'lr': 0.02, 'momentum': 0.9}, features, labels)
#增大了动量超参数后 目标函数值在后期迭代过程中的变化不够平滑
d2l.plt.show()

# 可以试着将学习率减小到原来的1/5。此时目标函数值在下降了一段时间后变化更加平滑
d2l.train_ch7(sgd_momentum, init_momentum_states(),
              {'lr': 0.004, 'momentum': 0.9}, features, labels)
d2l.plt.show()