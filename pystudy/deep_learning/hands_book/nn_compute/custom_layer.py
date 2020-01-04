"""
自定义层
    深度学习的一个魅力在于神经网络中各式各样的层，例如卷积层、池化层与循环层
    虽然Gluon提供了大量常用的层，但有时候我们依然希望自定义层。
    本节将介绍如何使用NDArray来自定义一个Gluon的层，从而可以被重复调用
"""
print("------------------------不含模型参数的自定义层¶----------------------")
# 跟使用Block类构造模型类似
# CenteredLayer类自定义了一个将输入减掉均值后输出的层，并将层的计算定义在了forward函数里。这个层里不含模型参数

from mxnet import gluon, nd
from mxnet.gluon import nn

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()

# 实例化层并做前向计算
layer = CenteredLayer()
"""
[-2. -1.  0.  1.  2.]
<NDArray 5 @cpu(0)>
"""
print(layer(nd.array([1, 2, 3, 4, 5])))

# 也可以用它来构造更复杂的模型
net = nn.Sequential()
net.add(nn.Dense(128),
        CenteredLayer())

# 打印自定义层各个输出的均值。因为均值是浮点数，所以它的值是一个很接近0的数
net.initialize()
y = net(nd.random.uniform(shape=(4, 8)))
print(y.mean().asscalar())  # -9.367795e-10


print("-----------------含模型参数的自定义层------------------")
# 模型参数可以通过训练学出
"""
在自定义含模型参数的层时，我们可以利用Block类自带的ParameterDict类型的成员变量params
它是一个由字符串类型的参数名字映射到Parameter类型的模型参数的字典
我们可以通过get函数从ParameterDict创建Parameter实例
"""
params = gluon.ParameterDict()
params.get('param2', shape=(2, 3))
"""
(
  Parameter param2 (shape=(2, 3), dtype=<class 'numpy.float32'>)
)
"""
print(params)

"""
现在我们尝试实现一个含权重参数和偏差参数的全连接层。
它使用ReLU函数作为激活函数。
其中in_units和units分别代表输入个数和输出个数
"""
class MyDense(nn.Block):
    # units为该层的输出个数，in_units为该层的输入个数
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)

# 我们实例化MyDense类并访问它的模型参数
dense = MyDense(units=3, in_units=5)
"""
mydense0_ (
  Parameter mydense0_weight (shape=(5, 3), dtype=<class 'numpy.float32'>)
  Parameter mydense0_bias (shape=(3,), dtype=<class 'numpy.float32'>)
)
"""
print(dense.params)

# 可以直接使用自定义层做前向计算
dense.initialize()
"""
[[0.06917784 0.01627153 0.01029644]
 [0.02602214 0.04537309 0.        ]]
<NDArray 2x3 @cpu(0)>
"""
print(dense(nd.random.uniform(shape=(2, 5))))


# 也可以使用自定义层构造模型
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
"""
[[0.03820475]
 [0.04035058]]
<NDArray 2x1 @cpu(0)>
"""
print(net(nd.random.uniform(shape=(2, 64))))