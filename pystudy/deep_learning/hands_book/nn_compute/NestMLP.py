from mxnet.gluon import nn
from mxnet import nd
from pystudy.hands_book.nn_compute import FancyMLP

"""
因为FancyMLP和Sequential类都是Block类的子类，所以我们可以嵌套调用它们
"""
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))

        self.dense = nn.Dense(16, activation='relu')

    def forward(self, x):
        return self.dense(self.net(x))

X = nd.random.uniform(shape=(2, 20))
net = nn.Sequential()
net.add(NestMLP(), nn.Dense(20), FancyMLP())
net.initialize()
net(X)