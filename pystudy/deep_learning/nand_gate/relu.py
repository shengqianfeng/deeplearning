
"""
ReLU函数只保留正数元素，并将负数元素清零
                                        ReLU(x)=max(x,0).

"""
import d2lzh as d2l
from mxnet import autograd, nd

def xyplot(x_vals, y_vals, name):
    d2l.set_figsize(figsize=(5, 2.5))
    d2l.plt.plot(x_vals.asnumpy(), y_vals.asnumpy())
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name + '(x)')
    d2l.plt.show()

x = nd.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = x.relu()    # 通过NDArray提供的relu函数来绘制ReLU函数
xyplot(x, y, 'relu')

"""
relu图像分析：
    当输入为负数时，ReLU函数的导数为0；当输入为正数时，ReLU函数的导数为1。
尽管输入为0时ReLU函数不可导，但是我们可以取此处的导数为0。
"""

# 绘制ReLU函数的导数。
y.backward()
xyplot(x, x.grad, 'grad of relu')