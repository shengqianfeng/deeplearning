from pystudy.nn_study.d2lzh import *

"""
线性回归从0开始的实现
"""
# 输入特征数为2
num_inputs = 2
# 样本个数1000
num_examples = 1000
# 权重
true_w = [2, -3.4]
# 偏置
true_b = 4.2
# features的每一行是一个长度为2的向量，一共1000行
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
print(features.shape)# (1000, 2)
print("--------------------------------")
# labels的每一行是一个长度为1的向量（标量）
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
"""
[[ 1.1630785   0.4838046 ]
 [ 0.29956347  0.15302546]
 [-1.1688148   1.558071  ]
 ...
 [-1.3481458   1.5419681 ]
 [-2.238252   -0.34891927]
 [ 0.02030763  1.0949801 ]]
<NDArray 1000x2 @cpu(0)>
"""
print(features)
"""
[1.1630785 0.4838046]
<NDArray 2 @cpu(0)>
"""
print(features[0])
"""
[4.879625]
<NDArray 1 @cpu(0)>
"""
print("--------------------------------")
print(labels)
"""
[4.879625]
<NDArray 1 @cpu(0)>
"""
print(labels[0])

def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

# set_figsize()
# plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1);  # 加分号只显示图

print("---------------------------")
# 读取第一个小批量数据样本并打印。每个批量的特征形状为(10, 2)，分别对应批量大小和输入个数；标签形状为批量大小
batch_size = 10
# for X, y in data_iter(batch_size, features, labels):
#     print(X, y)
#     break

# 初始化模型参数
# 将权重初始化成均值为0、标准差为0.01的正态随机数，偏差则初始化成0
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))

# 、之后的模型训练中，需要对这些参数求梯度来迭代参数的值，因此我们需要创建它们的梯度
w.attach_grad()
b.attach_grad()

lr = 0.03
num_epochs = 3
# 线性回归的矢量计算表达式  nd.dot(X, w) + b
net = linreg
# 损失函数
loss = squared_loss

for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
    # 和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)  # l是有关小批量X和y的损失
        l.backward()  # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().asnumpy()))

"""
epoch 1, loss 0.034896
epoch 2, loss 0.000122
epoch 3, loss 0.000048
    """
# 比较true_w和w