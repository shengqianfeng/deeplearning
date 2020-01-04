from pystudy.common.functions import *
"""
神经网络复盘之演进：三层神经网络

输入层：2个神经元
第一个隐藏层：3个神经元
第二个隐藏层：2个神经元
输出层：2个神经元
"""


"""
init_network()函数会进行权重和偏置的初始化，并将它们保存在字典变量network中。
这个字典变量 network中保存了每一层所需的参数（权重和偏置）。
"""
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])

    network['b3'] = np.array([0.1, 0.2])
    return network

# forward()函数中则封装了将输入信号转换为输出信号的处理过程
def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)    # [0.31682708 0.69627909]