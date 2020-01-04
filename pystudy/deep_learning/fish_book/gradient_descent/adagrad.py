import d2lzh as d2l
import math
from mxnet import nd

"""
AdaGrad算法对自变量的迭代轨迹
"""
def adagrad_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6  # 前两项为自变量梯度
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
d2l.plt.show()


# 将学习率增大到2。可以看到自变量更为迅速地逼近了最优解
eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
d2l.plt.show()


# 同动量法一样，AdaGrad算法需要对每个自变量维护同它一样形状的状态变量。
# 我们根据AdaGrad算法中的公式实现该算法
features, labels = d2l.get_data_ch7()

def init_adagrad_states():
    s_w = nd.zeros((features.shape[1], 1))
    s_b = nd.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s[:] += p.grad.square()
        p[:] -= hyperparams['lr'] * p.grad / (s + eps).sqrt()


d2l.train_ch7(adagrad, init_adagrad_states(), {'lr': 0.1}, features, labels)
d2l.plt.show()