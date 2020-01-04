"""
模型参数的访问、初始化和共享
"""
from mxnet import init, nd
from mxnet.gluon import nn

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()  # 使用默认初始化方式

X = nd.random.uniform(shape=(2, 20))
Y = net(X)  # 前向计算
print(Y) # (2,10)
print("-------------")
"""
模型参数的访问：
    1 对于使用Sequential类构造的神经网络，可以通过方括号[]来访问网络的任一层
    2 对于Sequential实例中含模型参数的层，我们可以通过Block类的params属性来访问该层包含的所有参数
"""
# 索引0表示隐藏层为Sequential实例最先添加的层
# 访问多层感知机net中隐藏层的所有参数。
"""
以下输出可以得到了一个由参数名称映射到参数实例的字典（类型为mxnet.gluon.parameter.ParameterDict类）
dense0_ (
  Parameter dense0_weight (shape=(256, 20), dtype=float32)  # 权重参数的名称为dense0_weight，它由net[0]的名称（dense0_）和自己的变量名（weight）组成。
  Parameter dense0_bias (shape=(256,), dtype=float32)
) <class 'mxnet.gluon.parameter.ParameterDict'>
"""
print(net[0].params, type(net[0].params))

# 为了访问特定参数，我们既可以通过名字来访问字典里的元素，也可以直接使用它的变量名
# Parameter dense0_weight (shape=(256, 20), dtype=float32) ------ Parameter dense0_weight (shape=(256, 20), dtype=float32)
print(net[0].params['dense0_weight'],   net[0].weight)


# Gluon里参数类型为Parameter类，它包含参数和梯度的数值，可以分别通过data函数和grad函数来访问
# 因为我们随机初始化了权重，所以权重参数是一个由随机数组成的形状为(256, 20)的NDArray
print(net[0].weight.data()) # (256,20)

# 权重梯度的形状和权重的形状一样。因为我们还没有进行反向传播计算，所以梯度的值全为0
print(net[0].weight.grad()) # # (256,20)

# 类似地，我们可以访问其他层的参数，如输出层的偏差值
"""
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
<NDArray 10 @cpu(0)>
"""
print(net[1].bias.data())
print("-------------------------------------------")
# 最后，我们可以使用collect_params函数来获取net变量所有嵌套（例如通过add函数嵌套）的层所包含的所有参数。
# 它返回的同样是一个由参数名称到参数实例的字典
"""
sequential0_ (
  Parameter dense0_weight (shape=(256, 20), dtype=float32)
  Parameter dense0_bias (shape=(256,), dtype=float32)
  Parameter dense1_weight (shape=(10, 256), dtype=float32)
  Parameter dense1_bias (shape=(10,), dtype=float32)
)
"""
print(net.collect_params())

# 这个collect_params函数可以通过正则表达式来匹配参数名，从而筛选需要的参数
"""
sequential0_ (
  Parameter dense0_weight (shape=(256, 20), dtype=float32)
  Parameter dense1_weight (shape=(10, 256), dtype=float32)
)
"""
print(net.collect_params('.*weight'))

print("--------------------------初始化权重------------------------------------")
# MXNet的init模块里提供了多种预设的初始化权重方法
# 将权重参数初始化成均值为0、标准差为0.01的正态分布随机数，并依然将偏差参数清零
# 非首次对模型初始化需要指定force_reinit为真
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
"""
[ 0.00195949 -0.0173764   0.00047347  0.00145809  0.00326049  0.00457878
 -0.00894258  0.00493839 -0.00904343 -0.01214079  0.02156406  0.01093822
  0.01827143 -0.0104467   0.01006219  0.0051742  -0.00806932  0.01376901
  0.00205885  0.00994352]
<NDArray 20 @cpu(0)>
"""
print(net[0].weight.data()[0])

# 使用常数来初始化权重参数
net.initialize(init=init.Constant(1), force_reinit=True)
"""
[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
<NDArray 20 @cpu(0)>
"""
print(net[0].weight.data()[0])

"""
如果只想对某个特定参数进行初始化，我们可以调用Parameter类的initialize函数，它与Block类提供的initialize函数的使用方法一致
"""
# 对隐藏层的权重使用Xavier随机初始化方法
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
"""
[ 0.00512482 -0.06579044 -0.10849719 -0.09586414  0.06394844  0.06029618
 -0.03065033 -0.01086642  0.01929168  0.1003869  -0.09339568 -0.08703034
 -0.10472868 -0.09879824 -0.00352201 -0.11063069 -0.04257748  0.06548801
  0.12987629 -0.13846186]
<NDArray 20 @cpu(0)>
"""
print(net[0].weight.data()[0])


print("-----------------自定义参数初始化方法------------------------------")
"""
 有时候我们需要的初始化方法并没有在init模块中提供。
 这时，可以实现一个Initializer类的子类，从而能够像使用其他初始化方法那样使用它。
 通常，我们只需要实现_init_weight这个函数，并将其传入的NDArray修改成初始化的结果
"""
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = nd.random.uniform(low=-10, high=10, shape=data.shape)
        data *= data.abs() >= 5

net.initialize(MyInit(), force_reinit=True)
"""
Init dense0_weight (256, 20)
Init dense1_weight (10, 256)
[-5.3659673  7.5773945  8.986376  -0.         8.827555   0.
  5.9840508 -0.         0.         0.         7.4857597 -0.
 -0.         6.8910007  6.9788704 -6.1131554  0.         5.4665203
 -9.735263   9.485172 ]
<NDArray 20 @cpu(0)>
"""
print(net[0].weight.data()[0])


# 我们还可以通过Parameter类的set_data函数来直接改写模型参数。例如，在下例中我们将隐藏层参数在现有的基础上加1
net[0].weight.set_data(net[0].weight.data() + 1)
"""
[-4.3659673  8.5773945  9.986376   1.         9.827555   1.
  6.9840508  1.         1.         1.         8.48576    1.
  1.         7.8910007  7.9788704 -5.1131554  1.         6.4665203
 -8.735263  10.485172 ]
<NDArray 20 @cpu(0)>
"""
print(net[0].weight.data()[0])


print("--------------共享模型参数--------------------------")
# 在有些情况下，我们希望在多个层之间共享模型参数
# 可以在Block类的forward函数里多次调用同一个层来计算
"""
另外一种方法，它在构造层的时候指定使用特定的参数。
如果不同层使用同一份参数，那么它们在前向计算和反向传播时都会共享相同的参数。
"""
# 我们让模型的第二隐藏层（shared变量）和第三隐藏层共享模型参数
net = nn.Sequential()
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,# 在构造第三隐藏层时通过params来指定它使用第二隐藏层的参数
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
# 因为模型参数里包含了梯度，所以在反向传播计算时，
# 第二隐藏层和第三隐藏层的梯度都会被累加在shared.params.grad()里
net.initialize()

X = nd.random.uniform(shape=(2, 20))
net(X)
"""
[1. 1. 1. 1. 1. 1. 1. 1.]
<NDArray 8 @cpu(0)>
"""
print(net[1].weight.data()[0] == net[2].weight.data()[0])
