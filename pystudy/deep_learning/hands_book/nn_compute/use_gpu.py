"""
GPU计算
将介绍如何使用单块NVIDIA GPU来计算。首先，需要确保已经安装好了至少一块NVIDIA GPU。
然后，下载CUDA并按照提示设置好相应的路径
这些准备工作都完成后，下面就可以通过nvidia-smi命令来查看显卡信息了
"""
# !nvidia-smi  # 对Linux/macOS用户有效

"""
计算设备：
    MXNet可以指定用来存储和计算的设备，如使用内存的CPU或者使用显存的GPU
    
    默认情况下，MXNet会将数据创建在内存，然后利用CPU来计算。
    在MXNet中，mx.cpu()（或者在括号里填任意整数）表示所有的物理CPU和内存。这意味着，MXNet的计算会尽量使用所有的CPU核
    
    但mx.gpu()只代表一块GPU和相应的显存。
    如果有多块GPU，我们用mx.gpu(i)来表示第i块GPU及相应的显存（i从0开始）且mx.gpu(0)和mx.gpu()等价。
"""
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn
# cpu(0)
# gpu(0)
# gpu(1)
print(mx.cpu())
print(mx.gpu())
print(mx.gpu(1))

print("---------NDArray的GPU计算-------")
# 在默认情况下，NDArray存在内存上。因此，之前我们每次打印NDArray的时候都会看到@cpu(0)这个标识
# 我们可以通过NDArray的context属性来查看该NDArray所在的设备
x = nd.array([1, 2, 3])
print(x.context)    # cpu(0)


print("--------GPU上的存储----------")
# 我们有多种方法将NDArray存储在显存上。例如，我们可以在创建NDArray的时候通过ctx参数指定存储设备
# 下面我们将NDArray变量a创建在gpu(0)上。注意，在打印a时，设备信息变成了@gpu(0)。
# 创建在显存上的NDArray只消耗同一块显卡的显存。
# 我们可以通过nvidia-smi命令查看显存的使用情况。
# 通常，我们需要确保不创建超过显存上限的数据。

a = nd.array([1, 2, 3], ctx=mx.gpu())
print(a)

# 假设至少有2块GPU，下面代码将会在gpu(1)上创建随机数组
B = nd.random.uniform(shape=(2, 3), ctx=mx.gpu(1))
print(B)

# 除了在创建时指定，我们也可以通过copyto函数和as_in_context函数在设备之间传输数据。
# 下面我们将内存上的NDArray变量x复制到gpu(0)上。

y = x.copyto(mx.gpu())
print(y)
z = x.as_in_context(mx.gpu())
print(z)

# 需要区分的是，如果源变量和目标变量的context一致，as_in_context函数使目标变量和源变量共享源变量的内存或显存
print(y.as_in_context(mx.gpu()) is y)
# copyto函数总是为目标变量开新的内存或显存
print(y.copyto(mx.gpu()) is y)

print("----------GPU上的计算-----------")
# MXNet的计算会在数据的context属性所指定的设备上执行。为了使用GPU计算，我们只需要事先将数据存储在显存上。计算结果会自动保存在同一块显卡的显存上。
print((z + 2).exp() * y)
"""
注意：
MXNet要求计算的所有输入数据都在内存或同一块显卡的显存上。这样设计的原因是CPU和不同的GPU之间的数据交互通常比较耗时。
因此，MXNet希望用户确切地指明计算的输入数据都在内存或同一块显卡的显存上。
例如，如果将内存上的NDArray变量x和显存上的NDArray变量y做运算，会出现错误信息。
当我们打印NDArray或将NDArray转换成NumPy格式时，如果数据不在内存里，MXNet会将它先复制到内存，从而造成额外的传输开销。
"""

print("-----------Gluon的GPU计算---------------")
# 同NDArray类似，Gluon的模型可以在初始化时通过ctx参数指定设备。
# 下面的代码将模型参数初始化在显存上
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=mx.gpu())
# 当输入是显存上的NDArray时，Gluon会在同一块显卡的显存上计算结果
print(net(y))

# 确认一下模型参数存储在同一块显卡的显存上
print(net[0].weight.data())




