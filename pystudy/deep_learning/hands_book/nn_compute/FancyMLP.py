from mxnet.gluon import nn
from mxnet import nd

"""
 构造复杂的网络，我们通过get_constant函数创建训练中不被迭代的参数，即常数参数
"""
class FancyMLP(nn.Block):

    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        # 使用get_constant创建的随机权重参数不会在训练中被迭代（即常数参数）
        self.rand_weight = self.params.get_constant('rand_weight', nd.random.uniform(shape=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, x):
        x = self.dense(x)
        # 使用创建的常数参数，以及NDArray的relu函数和dot函数
        x = nd.relu(nd.dot(x, self.rand_weight.data()) + 1)
        # 复用全连接层。等价于两个全连接层共享参数
        x = self.dense(x)
        # 控制流，这里我们需要调用asscalar函数来返回标量进行比较
        while x.norm().asscalar() > 1:
            x /= 2
        if x.norm().asscalar() < 0.8:
            x *= 10
        return x.sum()


"""
在这个FancyMLP模型中，我们使用了常数权重rand_weight（注意它不是模型参数）,
做了矩阵乘法操作（nd.dot）并重复使用了相同的Dense层.
下面我们来测试该模型的随机初始化和前向计算
"""
X = nd.random.uniform(shape=(2, 20))
net = FancyMLP()
net.initialize()
print(net(X))   # [19.304802]