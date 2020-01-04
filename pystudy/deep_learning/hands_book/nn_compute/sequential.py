from mxnet.gluon import nn
from mxnet import nd
"""
Sequential类的目的：
    它提供add函数来逐一添加串联的Block子类实例，而模型的前向计算就是将这些实例按添加的顺序逐一计算。

Sequential类的工作机制：
    实现与Sequential功能相同的MySequential类以帮助清晰地理解Sequential类的工作机制
"""
class MySequential(nn.Block):
    def __init__(self, **kwargs):
        super(MySequential, self).__init__(**kwargs)

    def add(self, block):
        # block是一个Block子类实例，假设它有一个独一无二的名字。我们将它保存在Block类的
        # 成员变量_children里，其类型是OrderedDict。
        # 当MySequential实例调用initialize函数时，系统会自动对_children里所有成员初始化
        self._children[block.name] = block

    def forward(self, x):
        # OrderedDict保证会按照成员添加时的顺序遍历成员
        for block in self._children.values():
            x = block(x)
        return x


X = nd.random.uniform(shape=(2, 20))
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
print(net(X))