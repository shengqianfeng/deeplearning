import d2lzh as d2l
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn

"""
多层感知机的简单实现
"""

# 定义模型
net = nn.Sequential()
# 多加了一个全连接层作为隐藏层。它的隐藏单元个数为256，并使用ReLU函数作为激活函数
net.add(nn.Dense(256, activation='relu'),nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))


# 读取数据并训练模型
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
              None, trainer)
