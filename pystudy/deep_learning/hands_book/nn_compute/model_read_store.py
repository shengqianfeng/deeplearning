"""
模型的读取和存储
    在实际中，我们有时需要把训练好的模型部署到很多不同的设备。
在这种情况下，我们可以把内存中训练好的模型参数存储在硬盘上供后续读取使用
"""
print("---------读写NDArray-------------------")
# 可以直接使用save函数和load函数分别存储和读取NDArray
# 创建了NDArray变量x，并将其存在文件名同为x的文件里
from mxnet import nd
from mxnet.gluon import nn

x = nd.ones(3)
nd.save('x', x)

# 然后我们将数据从存储的文件读回内存。
x2 = nd.load('x')
"""
[
[1. 1. 1.]
<NDArray 3 @cpu(0)>]
"""
print(x2)

# 还可以存储一列NDArray并读回内存
y = nd.zeros(4)
nd.save('xy', [x, y])
x2, y2 = nd.load('xy')
"""
(
[1. 1. 1.]
<NDArray 3 @cpu(0)>, 
[0. 0. 0. 0.]
<NDArray 4 @cpu(0)>)
"""
print((x2, y2))


# 甚至可以存储并读取一个从字符串映射到NDArray的字典
mydict = {'x': x, 'y': y}
nd.save('mydict', mydict)
mydict2 = nd.load('mydict')
"""
{'x': 
    [1. 1. 1.]
    <NDArray 3 @cpu(0)>, 
'y': 
    [0. 0. 0. 0.]
    <NDArray 4 @cpu(0)>
}
"""
print(mydict2)

print("--------读写Gluon模型的参数-------------")
# Gluon的Block类提供了save_parameters函数和load_parameters函数来读写模型参数
# 为了演示方便，我们先创建一个多层感知机，并将其初始化,由于延后初始化，我们需要先运行一次前向计算才能实际初始化模型参数
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = nd.random.uniform(shape=(2, 20))
Y = net(X)
# 下面把该模型的参数存成文件，文件名为mlp.params
filename = 'mlp.params'
net.save_parameters(filename)

# 接下来，我们再实例化一次定义好的多层感知机。与随机初始化模型参数不同，我们在这里直接读取保存在文件里的参数
net2 = MLP()
net2.load_parameters(filename)
# 因为这两个实例都有同样的模型参数，那么对同一个输入X的计算结果将会是一样的
Y2 = net2(X)
"""
[[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]
<NDArray 2x10 @cpu(0)>
"""
print(Y2 == Y)



