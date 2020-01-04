"""
填充和步幅
"""
from mxnet import nd
from mxnet.gluon import nn

# 定义一个函数来计算卷积层。它初始化卷积层权重，并对输入和输出做相应的升维和降维
def comp_conv2d(conv2d, X):
    conv2d.initialize()
    # (1, 1)代表批量大小和通道数（“多输入通道和多输出通道”一节将介绍）均为1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])  # 排除不关心的前两维：批量和通道

# 注意这里是两侧分别填充1行或列，所以在两侧一共填充2行或列
conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
X = nd.random.uniform(shape=(8, 8))
print(comp_conv2d(conv2d, X).shape) # (8, 8)

# 当卷积核的高和宽不同时，我们也可以通过设置高和宽上不同的填充数使输出和输入具有相同的高和宽
# 使用高为5、宽为3的卷积核。在高和宽两侧的填充数分别为2和1  公式：ph = kh−1  pw = kw−1
conv2d = nn.Conv2D(1, kernel_size=(5, 3), padding=(2, 1))
print(comp_conv2d(conv2d, X).shape)

print("------------------")
# 令高和宽上的步幅均为2，从而使输入的高和宽减半
conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
print(comp_conv2d(conv2d, X).shape) # (4, 4)

# 复杂一点 核心窗口为(3,5),步幅为(3,4)
conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
print(comp_conv2d(conv2d, X).shape)

