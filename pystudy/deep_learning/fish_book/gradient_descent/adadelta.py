"""
除了RMSProp算法以外，另一个常用优化算法AdaDelta算法也针对AdaGrad算法在
迭代后期可能较难找到有用解的问题做了改进,AdaDelta算法没有学习率这一超参数
"""
# AdaDelta算法需要对每个自变量维护两个状态变量，即 st 和 Δxt 。我们按AdaDelta算法中的公式实现该算法
import d2lzh as d2l
from mxnet import nd

features, labels = d2l.get_data_ch7()

def init_adadelta_states():
    s_w, s_b = nd.zeros((features.shape[1], 1)), nd.zeros(1)
    delta_w, delta_b = nd.zeros((features.shape[1], 1)), nd.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        s[:] = rho * s + (1 - rho) * p.grad.square()
        g = ((delta + eps).sqrt() / (s + eps).sqrt()) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g * g

# 使用超参数 ρ=0.9来训练模型
d2l.train_ch7(adadelta, init_adadelta_states(), {'rho': 0.9}, features, labels)
d2l.plt.show()


