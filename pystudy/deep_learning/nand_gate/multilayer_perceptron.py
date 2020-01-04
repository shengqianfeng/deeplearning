"""
多层感知机的实现
"""
import d2lzh as d2l
from mxnet import nd
from mxnet.gluon import loss as gloss

# 使用Fashion-MNIST数据集。我们将使用多层感知机对图像进行分类
batch_size = 256
# 获取读取数据
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# Fashion-MNIST数据集中图像形状为28×28，类别数为10
# 我们依然使用长度为28×28=784的向量表示每一张图像。
# 因此，输入个数为784，输出个数为10。实验中，我们设超参数隐藏单元个数为256
num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))
b1 = nd.zeros(num_hiddens)
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
b2 = nd.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()

# 定义激活函数
# 这里我们使用基础的maximum函数来实现ReLU，而非直接调用relu函数
def relu(X):
    return nd.maximum(X, 0)# 求最大值


# 通过reshape函数将每张原始图像改成长度为num_inputs的向量
# 定义模型
def net(X):
    X = X.reshape((-1, num_inputs)) # （256，784）
    H = relu(nd.dot(X, W1) + b1)
    return nd.dot(H, W2) + b2

# 直接使用Gluon提供的包括softmax运算和交叉熵损失计算的函数
# 定义损失函数
loss = gloss.SoftmaxCrossEntropyLoss()


# 训练模型
# 设超参数迭代周期数为5，学习率为0.5
num_epochs, lr = 5, 0.5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)




