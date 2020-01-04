from mxnet import autograd, nd

# 生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)



# 读取数据
from mxnet.gluon import data as gdata

batch_size = 10
# 将训练数据的特征和标签组合
dataset = gdata.ArrayDataset(features, labels)
# 随机读取小批量
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

# 定义模型
from mxnet.gluon import nn
# 在Gluon中，Sequential实例可以看作是一个串联各个层的容器。在构造模型时，我们在该容器中依次添加层。当给定输入数据时，容器中的每一层将依次计算并将输出作为下一层的输入
net = nn.Sequential()
# 作为一个单层神经网络，线性回归输出层中的神经元和输入层中各个输入完全连接。因此，线性回归的输出层又叫全连接层。在Gluon中，全连接层是一个Dense实例。我们定义该层输出个数为1
net.add(nn.Dense(1))
"""
值得一提的是，在Gluon中我们无须指定每一层输入的形状，例如线性回归的输入个数。当模型得到数据时，例如后面执行net(X)时，模型将自动推断出每一层的输入个数
"""

"""
初始化模型参数
在使用net前，我们需要初始化模型参数，如线性回归模型中的权重和偏差。我们从MXNet导入init模块。该模块提供了模型参数初始化的各种方法。这里的init是initializer的缩写形式。我们通过init.Normal(sigma=0.01)指定权重参数每个元素将在初始化时随机采样于均值为0、标准差为0.01的正态分布。偏差参数默认会初始化为零
"""
from mxnet import init

net.initialize(init.Normal(sigma=0.01))

"""
定义损失函数
用假名gloss代替导入的loss模块，并直接使用它提供的平方损失作为模型的损失函数
"""
from mxnet.gluon import loss as gloss

loss = gloss.L2Loss()  # 平方损失又称L2范数损失


"""
定义优化算法
创建一个Trainer实例，并指定学习率为0.03的小批量随机梯度下降（sgd）为优化算法。该优化算法将用来迭代net实例所有通过add函数嵌套的层所包含的全部参数。这些参数可以通过collect_params函数获取
"""
from mxnet import gluon

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})


"""
训练模型
在使用Gluon训练模型时，我们通过调用Trainer实例的step函数来迭代模型参数
"""
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))

"""
比较参数
分别比较学到的模型参数和真实的模型参数。我们从net获得需要的层，并访问其权重（weight）和偏差（bias）。学到的参数和真实的参数很接近
"""
dense = net[0]
"""
非常接近
[2, -3.4]
[[ 1.999684  -3.3991895]]
"""
print(true_w)
print(dense.weight.data())

print(true_b)
print(dense.bias.data())