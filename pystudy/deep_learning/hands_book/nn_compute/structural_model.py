"""
构造模型
    --->基于Block类的模型构造方法
        Block类是nn模块里提供的一个模型构造类，我们可以继承它来定义我们想要的模型
"""
from mxnet import nd
from mxnet.gluon import nn

# 继承Block类构造多层感知机
# MLP类重载了Block类的__init__函数和forward函数
class MLP(nn.Block):

    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):
        # 调用MLP父类Block的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        # 参数，如“模型参数的访问、初始化和共享”一节将介绍的模型参数params
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')  # 含有256个神经元的隐藏层
        self.output = nn.Dense(10)  # 含有10个神经元的输出层

    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    # 输入层-->隐藏层---->输出层
    def forward(self, x):
        return self.output(self.hidden(x))


# 20*2的联合分布
X = nd.random.uniform(shape=(2, 20))
net = MLP()
net.initialize()
# net(X)会调用MLP继承自Block类的__call__函数，这个函数将调用MLP类定义的forward函数来完成前向计算
# 结果是2*10的数组
print(net(X))
